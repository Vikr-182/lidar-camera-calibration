from lavis.models import load_model_and_preprocess
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna7b",
    is_eval=True,
    device="cuda:0",
)

print('Loading model done!')

use_nucleus_sampling = decoding_method == "Nucleus sampling"
image = vis_processors["eval"](image).unsqueeze(0).to(device)

samples = {
    "image": image,
    "prompt": prompt,
}

output = model.generate(
    samples,
    length_penalty=float(1),
    repetition_penalty=float(1),
    num_beams=5,
    max_length=150,
    min_length=1,
    top_p=0.2,
    use_nucleus_sampling=use_nucleus_sampling,
)

return output[0]