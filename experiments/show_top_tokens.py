"""Show CheXagent's top vocabulary tokens for the first answer word, next to p_present.

Repeats XRayVQATool._generate_response, but multiplies the final hidden state by the whole
lm_head (every vocabulary row, float32) instead of only the Yes/No rows.

    python experiments/show_top_tokens.py IMAGE [IMAGE ...] [--finding pneumonia] [--k 5]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402


def top_tokens(tool, image, prompt, k):
    tok, model = tool.tokenizer, tool.model
    query = tok.from_list_format([{"image": image}, {"text": prompt}])
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    input_ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(tool.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids, do_sample=False, num_beams=1, temperature=1.0, top_p=1.0, use_cache=True,
            max_new_tokens=8, return_dict_in_generate=True, output_hidden_states=True,
        )
        hidden = out.hidden_states[0][-1][0, -1]          # same vector p_present uses
        lm_head = model.get_output_embeddings()
        logits = hidden.float() @ lm_head.weight.float().T  # all rows, float32
        if lm_head.bias is not None:
            logits = logits + lm_head.bias.float()
        logits = logits[: len(tok)]                        # drop padding rows past the real vocab
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k)

    new_tokens = out.sequences[0][input_ids.size(1):]
    answer = tok.decode(new_tokens[:-1])
    yes_no = tool._yes_no_confidence(hidden, int(new_tokens[0]))
    return answer, yes_no, [(int(i), tok.decode([int(i)]), float(logits[i]), float(p))
                            for p, i in zip(top.values, top.indices)], lm_head.weight.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--finding", default="pneumonia")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    prompt = f"Is there {args.finding}?"
    for image in args.images:
        answer, yes_no, top, shape = top_tokens(tool, image, prompt, args.k)
        print(f"\n{Path(image).name}   prompt: {prompt!r}   lm_head: {tuple(shape)}, real vocab {len(tool.tokenizer)}")
        print(f"  answer: {answer!r}   p_present: {yes_no['p_yes'] if yes_no else None}")
        print(f"  {'rank':<5}{'id':>7}  {'token':<14}{'logit':>9}{'prob (full vocab)':>20}")
        for r, (i, s, lg, p) in enumerate(top, 1):
            print(f"  {r:<5}{i:>7}  {s!r:<14}{lg:>9.3f}{p:>20.4f}")


if __name__ == "__main__":
    main()
