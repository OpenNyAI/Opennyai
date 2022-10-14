class Record:
    def __init__(self, length, data):
        self.length = length
        self.data = data
    
    def to_tensor(self, pad_len):
        return self.data.to_tensor(pad_len)
    
    def get_data_length(self):
        return self.length
    
    def __str__(self):
        return "l:" + str(self.length)
    def __repr__(self):
        return "l:" + str(self.length)


class BucketBatch:
    
    @staticmethod
    def wrap_record(record):
        b = BucketBatch(max_record_len=record.get_data_length(), batch_size=1)
        b.add_record(record)
        return b
        
    def __init__(self, max_record_len, batch_size):
        self.max_record_len = max_record_len        
        self.batch_size = batch_size
        self.records = []
    
    def get_len(self):
        return len(self.records)
        
    def to_tensor(self, record_data_to_tensor_func, merge_records_func):
        result = None
        for r in self.records:
            r = record_data_to_tensor_func(r, self.max_record_len)
            result = merge_records_func(result, r)
        return result
    
    def is_full(self):
        '''Batch is full if no records cannot be inserted anymore without violating batch_size'''
        return (len(self.records) + 1) * self.max_record_len > self.batch_size
    
    def is_empty(self):
        return len(self.records) == 0
    
    def can_add_record(self, record):
        return not self.is_full() and record.length <= self.max_record_len
    
    def add_record(self, record):
        self.records.append(record)
    
    def pop_record(self):
        r = self.records.pop(0)
        return r
    
    def peek_record(self):
        return self.records[0]
    
    def __str__(self):
        return "{max_record_len:" + str(self.max_record_len) + ", length:" + str(len(self.records)) + ", records=" + str(self.records) + "}"
    def __repr__(self):
        return self.__str__()
    
    def take_records_from_bucket(self, bucket):
        while not self.is_full() and not bucket.is_empty():
            if self.can_add_record(bucket.peek_record()):
                self.add_record(bucket.pop_record())
            else:
                break
        
        

def bucket_records(records, batch_size):    
    len_to_buckets = dict()
    
    # group records with same length to buckets
    for r in records:
        record_len = r.length
        if record_len not in len_to_buckets:
            len_to_buckets[record_len] = [BucketBatch(record_len, batch_size)]
            
        bucket = len_to_buckets[record_len][-1]
        if not bucket.can_add_record(r):
            bucket = BucketBatch(record_len, batch_size)
            len_to_buckets[record_len].append(bucket)
        
        bucket.add_record(r)        
        
    # Flatten buckets into a list
    buckets_list = [b for sublist in len_to_buckets.values() for b in sublist]    
    
    # sort list of buckets: first come buckets with largest records 
    buckets_list = sorted(buckets_list, key = lambda b: b.max_record_len, reverse=True)
        
    # Merge buckets: move records of buckets with smaller records to buckets with larger records
    for i in range(0, len(buckets_list)):
        b = buckets_list[i]
                   
        if b.is_full() or b.is_empty():
            continue
        
        for j in range(i + 1, len(buckets_list)):
            nb = buckets_list[j]            
            
            if nb.is_empty():
                continue
                   
            b.take_records_from_bucket(nb)
                   
            if b.is_full():
                break            
            
    #filter empty buckets
    buckets_list = [b for b in buckets_list if not b.is_empty()]

    return buckets_list
