import csv
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import pipeline as hf_pipeline

from utils.evaluate_results import NO_ANSWER_MARKER, evaluate_results

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ===================== HARDWARE DETECTION & ADAPTIVE CONFIGURATION =====================
def get_device_config():
    """Detect hardware and return optimal configuration."""
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device = 'cuda'
        
        # Determine dtype and batch size based on GPU memory
        if gpu_memory_gb >= 16:
            dtype = torch.float32
            batch_size = 12
            logging.info(f"GPU detected: {gpu_name} ({gpu_memory_gb:.1f}GB) - Using float32, batch_size=12")
        elif gpu_memory_gb >= 8:
            dtype = torch.float16
            batch_size = 8
            logging.info(f"GPU detected: {gpu_name} ({gpu_memory_gb:.1f}GB) - Using float16, batch_size=8")
        else:
            dtype = torch.float16
            batch_size = 6
            logging.info(f"GPU detected: {gpu_name} ({gpu_memory_gb:.1f}GB) - Using float16, batch_size=6")
    else:
        # CPU only
        device = 'cpu'
        dtype = torch.float32
        batch_size = 2
        logging.info(f"No GPU detected - Using CPU only, dtype=float32, batch_size=2 (slower)")
    
    return {
        'device': device,
        'dtype': dtype,
        'batch_size': batch_size
    }

# Get optimal configuration for this PC
DEVICE_CONFIG = get_device_config()
DEVICE = DEVICE_CONFIG['device']
DTYPE = DEVICE_CONFIG['dtype']
BATCH_SIZE = DEVICE_CONFIG['batch_size']
HF_DEVICE_MAP = "auto" if DEVICE == "cuda" else "cpu"
QUESTION_LIMIT = 1000
logging.info(
    "Device config -> device=%s, dtype=%s, batch_size=%d, hf_device_map=%s",
    DEVICE,
    DTYPE,
    BATCH_SIZE,
    HF_DEVICE_MAP
)

model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE, device_map=HF_DEVICE_MAP, token=True)
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

PRIMARY_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,        # Greedy (Stable)
    pad_token_id=tokenizer.eos_token_id
)

SECONDARY_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=64,
    do_sample=True,
    temperature=0.2,        # Proven best temperature for secondary generation
    pad_token_id=tokenizer.eos_token_id
)

@dataclass
class QAExample:
    row: List[str]
    context: str
    question: str
    answer: str | None = None


def _build_messages(example: QAExample, system_prompt: str) -> List[Dict[str, str]]:
    user_prompt = f"""Context: {example.context} Question: {example.question} Answer:"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _decode_answer(response) -> str:
    if isinstance(response, list) and response:
        response = response[0]

    generated_text = response.get('generated_text')
    if isinstance(generated_text, list) and generated_text:
        answer = generated_text[-1].get('content', '')
    elif isinstance(generated_text, str):
        answer = generated_text
    else:
        answer = ''

    answer = answer.replace('\r', ' ').replace('\n', ' ').strip()
    if not answer or 'no answer' in answer.lower():
        return NO_ANSWER_MARKER
    return answer


def _normalize_answer(answer: str) -> str:
    return ' '.join(answer.lower().split())


def _merge_answers(primary: str, secondary: str, context: str) -> str:
    norm_primary = _normalize_answer(primary)
    norm_secondary = _normalize_answer(secondary)

    # 1. IMMEDIATE VETO: If ANY model predicts "NO ANSWER", default to NO ANSWER.
    if NO_ANSWER_MARKER.lower() in norm_primary or NO_ANSWER_MARKER.lower() in norm_secondary:
        return NO_ANSWER_MARKER

    # 2. CONTEXT VERIFICATION: Primary answer must appear verbatim in the context.
    context_lower = ' '.join(context.lower().split())
    primary_clean = ' '.join(primary.lower().split())

    if primary_clean in context_lower and len(primary) < 100:
        return primary

    # 3. Fallback: answer not found verbatim in context.
    return NO_ANSWER_MARKER


def _run_secondary_generation(batch: List[QAExample], system_prompt: str):
    """Use model.generate for the sampling pass so temperature is honored."""
    prompts = []
    for example in batch:
        messages = _build_messages(example, system_prompt)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=SECONDARY_GENERATION_CONFIG
        )

    prompt_token_count = inputs['input_ids'].shape[1]
    answers = []
    for idx in range(outputs.shape[0]):
        generated_tokens = outputs[idx, prompt_token_count:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer = answer.replace('\r', ' ').replace('\n', ' ').strip()
        if not answer or 'no answer' in answer.lower():
            answer = NO_ANSWER_MARKER
        answers.append(answer)
    return answers


def squad_qa(data_filename):
    """
    Query Llama model with SQuAD 2.0 questions and save answers to CSV.
    
    Reads CSV line by line (memory efficient) and processes each question:
    - Extracts context and question
    - Queries Llama-3.2-3B-Instruct model
    - Model returns answer if found in context, or "NO ANSWER" if not
    - Writes results incrementally to output CSV
    
    Args:
        data_filename: Path to CSV with columns: article, context, question, is_impossible, answers
        
    Returns:
        out_filename: Path to results CSV with added "final answer" column
    """

    # Ensure decoder-only batching pads on the left to prevent attention leakage
    tokenizer.padding_side = 'left'

    qa_pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=DTYPE,
        device_map=HF_DEVICE_MAP
    )

    # System prompt instructs the model on its task
    system_prompt = """You are a precise QA assistant.
Context: {provided passage}
Question: {user question}

Steps:
1. Search for the exact answer in the context.
2. If found, output ONLY the answer text.
3. If the answer is not explicitly stated, output exactly: NO ANSWER

Examples:
Context: The sky is blue.
Question: What color is the sky?
Answer: blue

Context: The sky is blue.
Question: Why is the sky blue?
Answer: NO ANSWER

Now answer."""
    
    # Prepare output filename
    out_filename = data_filename.replace('.csv', '-results.csv')
    
    # Determine total number of questions for progress reporting
    with open(data_filename, 'r', encoding='utf-8') as fin:
        total_questions = max(sum(1 for _ in fin) - 1, 1)  # subtract header

    target_questions = min(total_questions, QUESTION_LIMIT)

    print(f"Processing {target_questions}/{total_questions} questions from {data_filename}...")
    
    # Open input and output files
    with open(data_filename, 'r', newline='', encoding='utf-8') as fin, \
         open(out_filename, 'w', newline='', encoding='utf-8') as fout:
        
        csv_reader = csv.reader(fin)
        csv_writer = csv.writer(fout)
        
        # Read and write header
        header = next(csv_reader)
        context_idx = header.index('context')
        question_idx = header.index('question')
        output_header = header + ['final answer']
        csv_writer.writerow(output_header)
        
        context_groups: Dict[str, List[QAExample]] = defaultdict(list)
        all_rows: List[QAExample] = []

        for row in csv_reader:
            if len(all_rows) >= target_questions:
                break

            example = QAExample(
                row=row,
                context=row[context_idx],
                question=row[question_idx]
            )
            all_rows.append(example)
            context_groups[example.context].append(example)

        if context_groups:
            logging.info("Found %d unique contexts for %d questions", len(context_groups), len(all_rows))
            logging.info("Average %.1f questions per context", len(all_rows) / len(context_groups))

        processed = 0
        batch_items: List[QAExample] = []

        def flush_batch(batch: List[QAExample]):
            nonlocal processed
            if not batch or processed >= target_questions:
                return

            batch_messages = [_build_messages(item, system_prompt) for item in batch]
            primary_outputs = qa_pipe(batch_messages, generation_config=PRIMARY_GENERATION_CONFIG)
            secondary_answers = _run_secondary_generation(batch, system_prompt)

            for idx, item in enumerate(batch):
                if processed >= target_questions:
                    break

                primary_answer = _decode_answer(primary_outputs[idx])
                secondary_answer = secondary_answers[idx]
                item.answer = _merge_answers(primary_answer, secondary_answer, item.context) # with item.context
                processed += 1
                timestamp = datetime.now().strftime('%H:%M:%S')
                percent = (processed / target_questions) * 100
                print(f"[{timestamp}] Question {processed}/{target_questions} ({percent:5.1f}%)")

            batch.clear()

        for context, entries in context_groups.items():
            for entry in entries:
                batch_items.append(entry)
                if len(batch_items) >= BATCH_SIZE:
                    flush_batch(batch_items)

                if processed >= target_questions:
                    break

            if processed >= target_questions:
                break

        flush_batch(batch_items)

        for example in all_rows:
            answer = example.answer if example.answer is not None else NO_ANSWER_MARKER
            csv_writer.writerow(example.row + [answer])
    
    print(f'final answers recorded into {out_filename}')
    return out_filename


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    data = pd.read_csv(config['data'])
    sample = data.sample(n=config['sample_for_solution'])  # for grading will be replaced with 'sample_for_grading'
    sample_filename = config['data'].replace('.csv', '-sample.csv')
    sample.to_csv(sample_filename, index=False)

    out_filename = squad_qa(sample_filename)  # todo: the function you implement

    eval_out = evaluate_results(out_filename, final_answer_column='final answer')
    eval_out_list = [str((k, round(v, 3))) for (k, v) in eval_out.items()]
    print('\n'.join(eval_out_list))

    elapsed_time = time.time() - start_time
    print(f"time: {elapsed_time: .2f} sec")
