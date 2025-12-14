import onnxruntime_genai as rt

# Charger une seule fois au module
MODEL_PATH = "/home/leonardo/llm-models/mistral-onnx-int4"
model = rt.Model(MODEL_PATH)
tokenizer = rt.Tokenizer(model)


def generate_response(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"[{role.upper()}]: {content}\n"

    input_ids = tokenizer.encode(prompt)

    params = rt.GeneratorParams(model)
    params.set_model_input("input_ids", input_ids)
    params.set_search_options(temperature=0.7, top_p=0.9)

    generator = rt.Generator(model, params)

    while not generator.is_done():
        generator.generate_next_token()

    output_tokens = generator.get_output("output_ids")
    text = tokenizer.decode(output_tokens)

    return try_parse_json(text)
