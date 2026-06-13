import re

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from tqdm import tqdm

from .drag_modules import (
    PromptBuilderModule,
    QueryPoolModule,
    QueryStageDebateModule,
    debug_log,
    generate_single,
    render_messages,
    _preview,
)


ANSWER_PREFIX_RE = re.compile(r"\bthe\s+answer\s+is\s*:?", re.IGNORECASE)
SHORT_ANSWER_PREFIX_RE = re.compile(r"^(?:final\s+answer|answer)\s*:\s*", re.IGNORECASE)


class QueryDebateSingleAnswerRAG(BasicPipeline):
    """DRAG query debate followed by one retrieved-context answer generation call."""

    def __init__(
        self,
        config,
        prompt_template=None,
        max_query_debate_rounds=3,
        agents_num=2,
        query_proponent_agent=1,
        query_opponent_agent=1,
        generator=None,
        retriever=None,
        query_pool_module=None,
        prompt_builder_module=None,
        query_stage_module=None,
    ):
        super().__init__(config, prompt_template)
        self.config = config
        self.max_query_debate_rounds = max_query_debate_rounds

        if agents_num != query_proponent_agent + query_opponent_agent:
            raise ValueError(
                "The number of agents must be equal to the sum of query proponent and opponent agents"
            )
        if agents_num != 2:
            raise ValueError("The number of agents must be 2")
        if query_proponent_agent != 1 or query_opponent_agent != 1:
            raise ValueError(
                "DRAG_SINGLE requires exactly one query proponent agent and one query opponent agent"
            )
        if self.max_query_debate_rounds < 0:
            raise ValueError("max_query_debate_rounds must be greater than or equal to 0")

        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.agents_messages_query_stage = {}
        for i in range(query_proponent_agent):
            self.agents_messages_query_stage[f"Proponent Agent {i}"] = []
        for i in range(query_opponent_agent):
            self.agents_messages_query_stage[f"Opponent Agent {i}"] = []

        self.query_pool_module = (
            QueryPoolModule(self.retriever) if query_pool_module is None else query_pool_module
        )
        self.prompt_builder_module = (
            PromptBuilderModule(self.config, self.query_pool_module.format_query_pool)
            if prompt_builder_module is None
            else prompt_builder_module
        )
        self.query_stage_module = (
            QueryStageDebateModule(
                generator=self.generator,
                retriever=self.retriever,
                prompt_template=self.prompt_template,
                prompt_builder=self.prompt_builder_module,
                query_pool_module=self.query_pool_module,
                max_query_debate_rounds=self.max_query_debate_rounds,
                agents_messages_query_stage=self.agents_messages_query_stage,
            )
            if query_stage_module is None
            else query_stage_module
        )

    def run(self, dataset, do_eval=True):
        debug_log(
            self.config,
            f"Starting DRAG_SINGLE samples={len(dataset) if hasattr(dataset, '__len__') else 'unknown'} do_eval={do_eval}",
        )
        for idx, item in enumerate(tqdm(dataset, desc="Inference: ")):
            debug_log(self.config, f"Sample {idx} question='{_preview(item.question, self.config)}'")
            query_pool = self.query_stage_debate(item)
            item.update_output("QueryStage_QueryPool", query_pool)
            debug_log(self.config, f"Sample {idx} query_pool_size={len(query_pool)}")
            self.single_answer(item, query_pool)

        debug_log(self.config, "Calling FlashRAG evaluate/save")
        return self.evaluate(dataset, do_eval=do_eval)

    def query_stage_debate(self, item):
        return self.query_stage_module.run(item)

    def single_answer(self, item, query_pool):
        message = [
            self.prompt_builder_module.answer_only_message(query_pool),
            {"role": "user", "content": f"Question: {item.question}\n"},
        ]
        input_prompt = render_messages(message, self.config)
        debug_log(self.config, f"Answer prompt chars={len(input_prompt)}")
        output = generate_single(self.generator, input_prompt, self.config)
        parsed = self._parse_answer(output)
        item.update_output("answer_input_prompt", input_prompt)
        item.update_output("pred", parsed)
        item.update_output("raw_pred", output)
        debug_log(self.config, f"Final parsed answer='{_preview(parsed, self.config)}'")
        return output

    def format_query_pool(self, query_pool):
        return self.query_pool_module.format_query_pool(query_pool)

    def _parse_answer(self, output):
        text = "" if output is None else str(output).strip()
        matches = list(ANSWER_PREFIX_RE.finditer(text))
        if matches:
            text = text[matches[-1].end() :].strip()

        text = SHORT_ANSWER_PREFIX_RE.sub("", text).strip()
        text = text.strip("\"'`").strip()

        if self._is_strategyqa():
            match = re.match(r"^(yes|no)\b", text, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return text

    def _is_strategyqa(self):
        try:
            dataset_name = self.config["dataset_name"]
        except Exception:
            dataset_name = ""
        return str(dataset_name).lower() == "strategyqa"
