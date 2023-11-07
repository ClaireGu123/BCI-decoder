class ContrastiveBatch:
    def __init__(self, data):
        self.data = data
    def pin_memory(self,):
        self.data = self.data.pin_memory()
        return self
    
def contrastive_collate_wrapper(batch):
    return ContrastiveBatch(batch)