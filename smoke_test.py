import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

def main():
    print("=== Starting LLM Bridge 50-Question Smoke Test ===")
    
    # 1. Load HellaSwag validation dataset
    print("Loading HellaSwag validation dataset...")
    try:
        dataset = load_dataset("hellaswag", split="validation")
    except Exception as e:
        print(f"Error loading HellaSwag dataset: {e}. Exiting.")
        return

    # Select the first 50 validation samples
    num_samples = 50
    dataset_sample = dataset.select(range(num_samples))
    print(f"Loaded HellaSwag validation split. Selected first {len(dataset_sample)} samples.")

    # 2. Model configuration
    model_id = "EleutherAI/pythia-410m"
    print(f"Loading tokenizer and model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set device: MPS if available (macOS GPU), else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")
    
    # Load model
    model_dtype = torch.float32 if device in ["cpu", "mps"] else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    # 3. Setup Layer 12 forward hook to capture residual stream activations
    captured_acts = []
    
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured_acts.append(hidden_states.detach().cpu().to(torch.float16))

    # Hook the output of the 12th layer residual stream
    chosen_layer = 11  # layer 12 of 24
    print(f"Registering hook on gpt_neox.layers[{chosen_layer}]...")
    handle = model.gpt_neox.layers[chosen_layer].register_forward_hook(hook_fn)

    correct_predictions = 0
    all_scores = []
    
    print("\nEvaluating 50 questions zero-shot...")
    for idx in tqdm(range(num_samples)):
        sample = dataset_sample[idx]
        prompt = sample["ctx_a"] + " " + sample["ctx_b"]
        endings = sample["endings"]
        true_label = int(sample["label"])
        
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)
        
        # Limit prompt size
        max_seq_len = 128
        if prompt_len > max_seq_len:
            prompt_ids = prompt_ids[-max_seq_len:]
            prompt_len = max_seq_len

        # Extract activations
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        captured_acts.clear()
        
        with torch.no_grad():
            model(prompt_tensor)
            
        assert len(captured_acts) == 1, "Hook failed to capture activations."
        prompt_activations = captured_acts[0]
        # Shape verification for the first sample
        if idx == 0:
            print(f"Sample 0 prompt activations shape: {prompt_activations.shape}")
            assert prompt_activations.shape[-1] == 1024, f"Expected d_model=1024, got {prompt_activations.shape[-1]}"
            assert prompt_activations.shape[1] == len(prompt_ids), "Sequence length mismatch."

        # Evaluate endings zero-shot using length-normalized log-likelihood
        ending_scores = []
        for ending in endings:
            ending_clean = " " + ending.strip()
            ending_ids = tokenizer.encode(ending_clean, add_special_tokens=False)
            
            full_ids = prompt_ids + ending_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            captured_acts.clear()
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits
                
            shift_logits = logits[0, prompt_len-1 : -1, :]
            shift_labels = input_tensor[0, prompt_len:]
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = log_probs[torch.arange(len(ending_ids)), shift_labels]
            
            score = target_log_probs.mean().item()
            ending_scores.append(score)

        predicted_label = int(np.argmax(ending_scores))
        if predicted_label == true_label:
            correct_predictions += 1
            
    # Remove the forward hook
    handle.remove()
    print("Hook removed successfully.")

    # Calculate and verify accuracy
    accuracy = (correct_predictions / num_samples) * 100
    print(f"\n=== Zero-Shot Performance Summary ===")
    print(f"Model: {model_id}")
    print(f"Total evaluated: {num_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"=====================================")

    if 40.0 <= accuracy <= 50.0:
        print("Success! Accuracy is in the expected 40–50% range.")
        print("Pythia-410M is confirmed as the correct baseline model.")
    else:
        print(f"Warning: Accuracy {accuracy:.2f}% is outside the expected 40-50% sweet spot.")
        print("Please note this result. If it deviates significantly in a full run, we will re-baseline.")

if __name__ == "__main__":
    main()
