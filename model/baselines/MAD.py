from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever


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


def _safe_generate_batch(generator, prompts, config):
    if config.get("framework") != "openai":
        return generator.generate(prompts)

    conversations = [_normalize_openai_conversation(prompt) for prompt in prompts]
    if len(conversations) == 1:
        # Work around flashrag OpenAI generator single-item shape bug.
        duplicated = [conversations[0], conversations[0]]
        duplicated_output = generator.generate(duplicated)
        return [duplicated_output[0]]

    return generator.generate(conversations)


class MultiAgentDebate(BasicPipeline):
    def __init__(
        self,
        config,
        prompt_template=None,
        debate_rounds=3,
        agents_num=2,
        rag_agents_num=0,
        generator=None,
        retriever=None,
    ):
        super().__init__(config, prompt_template)
        self.config = config
        self.debate_rounds = debate_rounds
        # If use rag, the number of normal agents is reduced by rag agents.
        self.agents_num = agents_num - rag_agents_num

        self.generator = get_generator(config) if generator is None else generator
        self.retriever = None
        if rag_agents_num > 0:
            self.retriever = get_retriever(config) if retriever is None else retriever

        self.agents_messages = dict()
        # Initialize the agents' messages
        self.rag_agents_num = rag_agents_num
        if self.rag_agents_num > 0:
            for i in range(self.rag_agents_num):
                self.agents_messages[f"RAG Agent {i + 1}"] = dict()
        for i in range(self.agents_num):
            self.agents_messages[f"Agent {i + 1}"] = dict()

    def run(self, dataset, do_eval=True):
        for round_id in range(self.debate_rounds):
            for agent_name in self.agents_messages:
                if round_id == 0:
                    # Construct initial prompt for each agent.
                    if "RAG" in agent_name:
                        input_query = dataset.question
                        retrieval_results = self.retriever.batch_search(input_query)
                        dataset.update_output("retrieval_result", retrieval_results)
                        round_messages = [
                            [
                                self._construct_rag_agents_system_message(
                                    self._format_reference(r)
                                ),
                                {"role": "user", "content": f"Question: {q}"},
                            ]
                            for q, r in zip(dataset.question, dataset.retrieval_result)
                        ]
                    else:
                        round_messages = [
                            [
                                self._construct_agents_system_message(),
                                {"role": "user", "content": f"Question: {q}"},
                            ]
                            for q in dataset.question
                        ]

                    self.agents_messages[agent_name] = round_messages
                    input_prompt = [
                        self.prompt_template.get_string(messages=round_message)
                        for round_message in round_messages
                    ]
                    output = _safe_generate_batch(self.generator, input_prompt, self.config)

                    dataset.update_output(
                        f"{agent_name}_Round_{round_id}_input_prompt", input_prompt
                    )
                    dataset.update_output(f"{agent_name}_Round_{round_id}_output", output)
                else:
                    for q_id, question in enumerate(dataset.question):
                        other_agents = {
                            k: v for k, v in self.agents_messages.items() if k != agent_name
                        }
                        debate_message = self._construct_debate_user_message(
                            other_agents,
                            question,
                            q_id,
                            round_id,
                        )
                        self.agents_messages[agent_name][q_id].append(debate_message)

                    input_prompt = [
                        self.prompt_template.get_string(messages=agent_message)
                        for agent_message in self.agents_messages[agent_name]
                    ]
                    output = _safe_generate_batch(self.generator, input_prompt, self.config)

                    dataset.update_output(
                        f"{agent_name}_Round_{round_id}_input_prompt", input_prompt
                    )
                    dataset.update_output(f"{agent_name}_Round_{round_id}_output", output)

                for idx, message in enumerate(output):
                    self.agents_messages[agent_name][idx].append(
                        {"role": "assistant", "content": message}
                    )

            agents_responses = []
            for q_idx in range(len(dataset.question)):
                response_text = "Agents responses:\n"
                for agent_name in self.agents_messages:
                    response_text += (
                        f"{agent_name}: {self.agents_messages[agent_name][q_idx][-1]['content']}\n"
                    )
                agents_responses.append(response_text)

            moderator_messages = [
                [
                    self._construct_moderator_system_message(),
                    {"role": "user", "content": f"Question: {q}\n{agents_responses[i]}"},
                ]
                for i, q in enumerate(dataset.question)
            ]
            moderator_input_prompt = [
                self.prompt_template.get_string(messages=moderator_message)
                for moderator_message in moderator_messages
            ]
            moderator_output = _safe_generate_batch(self.generator, moderator_input_prompt, self.config)

            dataset.update_output(f"Moderator_Round_{round_id}_input_prompt", moderator_input_prompt)
            dataset.update_output(f"Moderator_Round_{round_id}_output", moderator_output)
            dataset.update_output("pred", moderator_output)

        dataset = self.evaluate(dataset, do_eval=do_eval)
        return dataset

    def _construct_agents_system_message(self):
        if self.config["dataset_name"] == "StrategyQA":
            return {
                "role": "system",
                "content": (
                    "Answer the question based on your own knowledge. Given two answer candidates, "
                    "Yes and No, choose the best answer choice. Explain your answer, and always put "
                    "the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response."
                ),
            }
        return {
            "role": "system",
            "content": (
                "Answer the question based on your own knowledge. Explain your answer, and always put "
                "the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response."
            ),
        }

    def _construct_rag_agents_system_message(self, retrieval_results):
        if self.config["dataset_name"] == "StrategyQA":
            return {
                "role": "system",
                "content": (
                    "Answer the question based on the given document. Given two answer candidates, "
                    "Yes and No, choose the best answer choice. Explain your answer, and always put "
                    "the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your "
                    "response. The following are given documents.\n"
                    f"{retrieval_results}"
                ),
            }
        return {
            "role": "system",
            "content": (
                "Answer the question based on the given document. Explain your answer, and always put "
                "the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your "
                "response. The following are given documents.\n"
                f"{retrieval_results}"
            ),
        }

    def _construct_debate_user_message(self, other_agents, question, question_id, round_id):
        if self.config["dataset_name"] == "StrategyQA":
            debate_messages = (
                "I will give the answers and arguments to this question from other agents. Use their "
                "solution as additional advice; note that they may be wrong. Given two answer candidates, "
                "Yes and No, choose the best answer choice. Explain your answer, and always put the answer "
                "after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response."
            )
        else:
            debate_messages = (
                "I will give the answers and arguments to this question from other agents. Use their "
                "solution as additional advice; note that they may be wrong. Explain your answer, and always "
                "put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response."
            )
        debate_messages += f"Question: {question}\n"
        debate_messages += "Other agents responses:\n"
        for agent_name in other_agents:
            debate_messages += (
                f"{agent_name}: {other_agents[agent_name][question_id][round_id * 2]['content']}\n"
            )

        return {"role": "user", "content": debate_messages}

    def _construct_moderator_system_message(self):
        if self.config["dataset_name"] == "StrategyQA":
            return {
                "role": "system",
                "content": (
                    "You are a moderator in a debate competition. Your task is to determine the correct "
                    "final answer based on the arguments presented by debaters. Given two answer candidates, "
                    "Yes and No, choose the best answer choice. Output only the final answer with no "
                    "explanations or additional text."
                ),
            }
        return {
            "role": "system",
            "content": (
                "You are a moderator in a debate competition. Your task is to determine the correct final "
                "answer based on the arguments presented by debaters. Output only the final answer with no "
                "explanations or additional text."
            ),
        }

    def _format_reference(self, retrieval_result):
        formatted_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            formatted_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return formatted_reference
