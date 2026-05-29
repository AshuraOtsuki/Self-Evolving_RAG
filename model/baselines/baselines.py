from flashrag.prompt import PromptTemplate
from flashrag.pipeline import BasicPipeline


def _normalize_openai_conversation(prompt):
    if isinstance(prompt, list):
        if len(prompt) == 0:
            text = ""
        elif all(isinstance(item, dict) for item in prompt):
            text = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in prompt
            )
        else:
            text = "\n".join(str(item) for item in prompt)
    elif isinstance(prompt, dict):
        text = prompt.get("content", str(prompt))
    else:
        text = str(prompt)
    return [{"role": "user", "content": text}]


def _safe_generate_single(generator, prompt, config):
    if config["framework"] == "openai":
        conversation = _normalize_openai_conversation(prompt)
        return generator.generate(conversation)[0]
    return generator.generate(prompt)[0]


def single_agent_pred_parse(dataset):
    final_answer_prefix = "The answer is:"
    for item in dataset:
        pred = item.pred
        if final_answer_prefix in pred:
            answer = pred.split(final_answer_prefix)[1].strip()
        else:
            answer = pred
        item.update_output("raw_pred", pred)
        item.update_output("pred", answer)
    return dataset


single_agent_prompt_template = {
    "StrategyQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on your own knowledge. "
            "Given four answer candidates, Yes and No, choose the best answer choice. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', "
            "at the end of your response. "
        ),
        user_prompt="Question: {question}",
    ),
    "NQ": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on your own knowledge. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
        ),
        user_prompt="Question: {question}",
    ),
    "TriviaQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on your own knowledge. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
        ),
        user_prompt="Question: {question}",
    ),
    "HotpotQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on your own knowledge. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
        ),
        user_prompt="Question: {question}",
    ),
    "2wiki": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on your own knowledge. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
        ),
        user_prompt="Question: {question}",
    ),
    "PopQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on your own knowledge. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
        ),
        user_prompt="Question: {question}",
    ),
}


standard_rag_prompt_template = {
    "StrategyQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on the given document. "
            "Given four answer candidates, Yes and No, choose the best answer choice. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', "
            "at the end of your response. "
            "\nThe following are given documents.\n{reference}"
        ),
        user_prompt="Question: {question}",
    ),
    "NQ": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on the given document. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
            "\nThe following are given documents.\n{reference}"
        ),
        user_prompt="Question: {question}",
    ),
    "TriviaQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on the given document. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
            "\nThe following are given documents.\n{reference}"
        ),
        user_prompt="Question: {question}",
    ),
    "HotpotQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on the given document. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
            "\nThe following are given documents.\n{reference}"
        ),
        user_prompt="Question: {question}",
    ),
    "2wiki": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on the given document. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
            "\nThe following are given documents.\n{reference}"
        ),
        user_prompt="Question: {question}",
    ),
    "PopQA": lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=(
            "Answer the question based on the given document. "
            "Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', "
            "at the end of your response. "
            "\nThe following are given documents.\n{reference}"
        ),
        user_prompt="Question: {question}",
    ),
}


class Baselines(BasicPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def run(self, method_name, test_data, do_eval=False):
        method_map = {
            "naive_gen": self.naive_gen,
            "naive_rag": self.naive_rag,
            "flare": self.flare,
            "iterretgen": self.iterretgen,
            "ircot": self.ircot,
            "self_ask": self.self_ask,
            "sure": self.sure,
            "selfrag": self.selfrag,
            "retrobust": self.retrobust,
            "mad": self.mad,
        }
        if method_name not in method_map:
            raise ValueError(f"Unsupported baseline method: {method_name}")
        if method_name == "naive_gen":
            return method_map[method_name](test_data, do_eval=do_eval)
        return method_map[method_name](test_data)

    def naive_gen(self, test_data, do_eval=False):
        from flashrag.utils import get_generator

        template = single_agent_prompt_template[self.config["dataset_name"]](self.config)
        generator = get_generator(self.config)

        for item in test_data:
            input_prompt = template.get_string(question=item.question)
            output = _safe_generate_single(generator, input_prompt, self.config)
            item.update_output("input_prompt", input_prompt)
            item.update_output("pred", output)

        parsed_data = single_agent_pred_parse(test_data)
        return self.evaluate(parsed_data, do_eval=do_eval)

    def naive_rag(self, test_data):
        from flashrag.pipeline import SequentialPipeline

        template = standard_rag_prompt_template[self.config["dataset_name"]](self.config)
        pipeline = SequentialPipeline(self.config, template)
        return pipeline.run(test_data, pred_process_fun=single_agent_pred_parse)

    def flare(self, test_data):
        """
        Reference:
            Zhengbao Jiang et al. "Active Retrieval Augmented Generation"
            in EMNLP 2023.
        """
        from flashrag.pipeline import FLAREPipeline

        pipeline = FLAREPipeline(self.config)
        return pipeline.run(test_data)

    def iterretgen(self, test_data):
        """
        Reference:
            Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with
                                Iterative Retrieval-Generation Synergy"
            in EMNLP Findings 2023.

            Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
            in EMNLP Findings 2023.
        """
        from flashrag.pipeline import IterativePipeline

        iter_num = 3
        template = standard_rag_prompt_template[self.config["dataset_name"]](self.config)
        pipeline = IterativePipeline(self.config, prompt_template=template, iter_num=iter_num)
        return pipeline.run(test_data, pred_process_fun=single_agent_pred_parse)

    def ircot(self, test_data):
        """
        Reference:
            Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning
            for Knowledge-Intensive Multi-Step Questions" in ACL 2023
        """
        from flashrag.pipeline import IRCOTPipeline

        pipeline = IRCOTPipeline(self.config, max_iter=self.config["max_answer_debate_rounds"])
        return pipeline.run(test_data)

    def self_ask(self, test_data):
        """
        Reference:
            Ofir Press et al. "Measuring and Narrowing the Compositionality Gap in Language
            Models" in EMNLP Findings 2023.
        """
        from flashrag.pipeline import SelfAskPipeline

        if self.config["dataset_name"] in ["StrategyQA", "HotpotQA", "2wiki"]:
            pipeline = SelfAskPipeline(self.config, max_iter=5, single_hop=True)
            return pipeline.run(test_data)
        if self.config["dataset_name"] in ["NQ", "PopQA", "TriviaQA"]:
            pipeline = SelfAskPipeline(self.config, max_iter=5, single_hop=False)
            return pipeline.run(test_data)
        raise ValueError("Dataset not supported")

    def sure(self, test_data):
        """
        Reference:
            Jaehyung Kim et al. "SuRe: Summarizing Retrievals using Answer Candidates for
            Open-domain QA of LLMs" in ICLR 2024
            Official repo: https://github.com/bbuing9/ICLR24_SuRe
        """
        from flashrag.pipeline import SuRePipeline

        pipeline = SuRePipeline(self.config)
        return pipeline.run(test_data)

    def selfrag(self, test_data):
        """
        Reference:
            Akari Asai et al. "SELF-RAG: Learning to Retrieve, Generate and Critique through
            self-reflection" in ICLR 2024.
            Official repo: https://github.com/AkariAsai/self-rag
        """
        from flashrag.pipeline import SelfRAGPipeline

        pipeline = SelfRAGPipeline(
            self.config,
            threshold=0.2,
            max_depth=2,
            beam_width=2,
            w_rel=1.0,
            w_sup=1.0,
            w_use=1.0,
            use_grounding=True,
            use_utility=True,
            use_seqscore=True,
            ignore_cont=True,
            mode="adaptive_retrieval",
        )
        return pipeline.run(test_data, long_form=False)

    def retrobust(self, test_data):
        """
        Reference:
            Ori Yoran et al. "Making Retrieval-Augmented Language Models Robust to Irrelevant
            Context" in ICLR 2024.
            Official repo: https://github.com/oriyor/ret-robust
        """
        from flashrag.pipeline import SelfAskPipeline
        from flashrag.utils import selfask_pred_parse

        model_dict = {
            "nq": "/data/share/baseline_ckpt/Ret_Robust/llama-2-13b-peft-nq-retrobust",
            "2wiki": "/data/share/baseline_ckpt/Ret_Robust/llama-2-13b-peft-2wikihop-retrobust",
        }
        if self.config["dataset_name"] in ["NQ", "TriviaQA", "PopQA"]:
            lora_path = model_dict["nq"]
        elif self.config["dataset_name"] in ["HotpotQA", "2wiki", "StrategyQA"]:
            lora_path = model_dict["2wiki"]
        else:
            print("Not use lora")
            lora_path = model_dict.get(self.config["dataset_name"], None)

        self.config["generator_lora_path"] = lora_path
        pipeline = SelfAskPipeline(self.config, max_iter=5, single_hop=False)
        return pipeline.run(test_data, pred_process_fun=selfask_pred_parse)

    def mad(self, test_data):
        from .MAD import MultiAgentDebate

        pipeline = MultiAgentDebate(
            self.config,
            debate_rounds=self.config["max_answer_debate_rounds"],
            agents_num=self.config["agents"],
            rag_agents_num=self.config["rag_agents"],
        )
        return pipeline.run(test_data)


def naive_gen(cfg, test_data):
    return Baselines(cfg).naive_gen(test_data)


def naive_rag(cfg, test_data):
    return Baselines(cfg).naive_rag(test_data)


def flare(cfg, test_data):
    return Baselines(cfg).flare(test_data)


def iterretgen(cfg, test_data):
    return Baselines(cfg).iterretgen(test_data)


def ircot(cfg, test_data):
    return Baselines(cfg).ircot(test_data)


def self_ask(cfg, test_data):
    return Baselines(cfg).self_ask(test_data)


def sure(cfg, test_data):
    return Baselines(cfg).sure(test_data)


def selfrag(cfg, test_data):
    return Baselines(cfg).selfrag(test_data)


def retrobust(cfg, test_data):
    return Baselines(cfg).retrobust(test_data)


def mad(cfg, test_data):
    return Baselines(cfg).mad(test_data)
