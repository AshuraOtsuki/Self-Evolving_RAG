import string

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from tqdm import tqdm

from .drag_modules import (
    AnswerStageDebateModule,
    PromptBuilderModule,
    QueryPoolModule,
    QueryStageDebateModule,
)


class DebateAugmentedRAG(BasicPipeline):
    def __init__(
        self,
        config,
        prompt_template=None,
        max_query_debate_rounds=3,
        max_answer_debate_rounds=3,
        agents_num=2,
        query_proponent_agent=1,
        query_opponent_agent=1,
        answer_proponent_agent=1,
        answer_opponent_agent=1,
        generator=None,
        retriever=None,
        query_pool_module=None,
        prompt_builder_module=None,
        query_stage_module=None,
        answer_stage_module=None,
    ):
        super().__init__(config, prompt_template)
        self.config = config
        self.max_query_debate_rounds = max_query_debate_rounds
        self.max_answer_debate_rounds = max_answer_debate_rounds

        if (
            agents_num != query_proponent_agent + query_opponent_agent
            & agents_num != answer_proponent_agent + answer_opponent_agent
        ):
            raise ValueError(
                "The number of agents must be equal to the sum of the proponent and opponent agents"
            )
        if agents_num != 2:
            raise ValueError("The number of agents must be 2")

        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.agents_messages_answer_stage = dict()
        self.agents_messages_query_stage = dict()
        # Initialize the agents' messages
        for i in range(query_proponent_agent):
            self.agents_messages_query_stage[f"Proponent Agent {i}"] = []
        for i in range(query_opponent_agent):
            self.agents_messages_query_stage[f"Opponent Agent {i}"] = []

        for i in range(answer_proponent_agent):
            self.agents_messages_answer_stage[f"Proponent Agent {i}"] = []
        for i in range(answer_opponent_agent):
            self.agents_messages_answer_stage[f"Opponent Agent {i}"] = []

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
        self.answer_stage_module = (
            AnswerStageDebateModule(
                generator=self.generator,
                prompt_template=self.prompt_template,
                prompt_builder=self.prompt_builder_module,
                query_pool_module=self.query_pool_module,
                max_answer_debate_rounds=self.max_answer_debate_rounds,
                agents_messages_answer_stage=self.agents_messages_answer_stage,
            )
            if answer_stage_module is None
            else answer_stage_module
        )

    def run(self, dataset, do_eval=True):
        for item in tqdm(dataset, desc="Inference: "):
            query_pool = self.query_stage_debate(item)
            item.update_output("QueryStage_QueryPool", query_pool)

            # If answer debate rounds is greater than 0, then answer the question with answer debate
            if self.max_answer_debate_rounds > 0:
                self.answer_stage_debate(item, query_pool)
            else:  # If answer debate rounds is 0, then only answer the question, no debate
                message = [
                    self._answer_only_message(query_pool),
                    {"role": "user", "content": f"Question: {item.question}\n"},
                ]
                input_prompt = self.prompt_template.get_string(messages=message)
                output = self.generator.generate(input_prompt)[0]
                item.update_output("answer_input_prompt", input_prompt)
                item.update_output("pred", output)

        dataset = self.evaluate(dataset, do_eval=do_eval)

    def query_stage_debate(self, item):
        return self.query_stage_module.run(item)

    def answer_stage_debate(self, item, query_pool):
        return self.answer_stage_module.run(item, query_pool)

    def _query_stage_system_message(self, agent_name):
        return self.prompt_builder_module.query_stage_system_message(agent_name)

    def _query_stage_moderator_message(self, agents_messages, input_query, query_pool):
        return self.prompt_builder_module.query_stage_moderator_message(
            agents_messages, input_query, query_pool
        )

    def _answer_stage_system_message(self, agent_name, query_pool):
        return self.prompt_builder_module.answer_stage_system_message(agent_name, query_pool)

    def _answer_only_message(self, query_pool):
        return self.prompt_builder_module.answer_only_message(query_pool)

    def _answer_stage_debate_message(self, other_agents, question, round):
        return self.prompt_builder_module.answer_stage_debate_message(other_agents, question, round)

    def _answer_stage_moderator_message(self):
        return self.prompt_builder_module.answer_stage_moderator_message()

    def _format_reference(self, retrieval_result):
        return self.query_pool_module.format_reference(retrieval_result)

    def maintain_query_pool(self, query_pool, opponent_output):
        return self.query_pool_module.maintain_query_pool(query_pool, opponent_output)

    def format_query_pool(self, query_pool):
        return self.query_pool_module.format_query_pool(query_pool)

    def find_most_similar_key(self, query_dict, target_query):
        return self.query_pool_module.find_most_similar_key(query_dict, target_query)
