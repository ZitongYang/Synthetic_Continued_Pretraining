from vllm import LLM, SamplingParams
import gradio as gr

class ChatBot:
    def __init__(self, model, tokenizer, sampling_params):
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.dialog = [
            {"role": "system", "content": "You are a helpful AI assistant for responding to user instructions"},
        ]
    
    def query(self, prompt: str):
        self.dialog.append({"role": "user", "content": prompt})
        full_prompt = self.tokenizer.apply_chat_template(self.dialog, tokenize=False)
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        output = self.model.generate(full_prompt, self.sampling_params)
        response = output[0].outputs[0].text
        self.dialog.append({"role": "assistant", "content": response})
        return response


if __name__ == "__main__":
    # Initialize the model
    model = LLM(model="ckpts/instruct-lr5e-06-rr0.1-epochs2-bs128-wd0.01-warmup0.05-qualitylr5e06rr0.1epochs2bs16wd0.01warmup0.05MetaLlama38B",
                tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct",
                tensor_parallel_size=8, device="cuda")
    tokenizer = model.get_tokenizer()
    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=0.1, max_tokens=2000, stop=[tokenizer.eos_token])

    def chatbot_fn(prompt):
        chatbot = ChatBot(model, tokenizer, sampling_params)
        return chatbot.query(prompt)
    demo = gr.Interface(fn=chatbot_fn, inputs="textbox", outputs="textbox")    
    demo.launch(share=True)