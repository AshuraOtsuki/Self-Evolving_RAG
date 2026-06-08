class LessonRetriever:
    def __init__(self, memory_bank, top_k=5, min_score=0.15):
        self.memory_bank = memory_bank
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question, entity_or_topic=None):
        if self.memory_bank is None:
            return []
        return self.memory_bank.retrieve_lessons(
            question,
            entity_or_topic=entity_or_topic,
            top_k=self.top_k,
            min_score=self.min_score,
        )
