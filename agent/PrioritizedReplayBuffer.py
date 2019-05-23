from collections import namedtuple
#from random import sample as randsample
import numpy as np
import torch


from agent.sumtree import SumTree



Transition = namedtuple("Transition", ("state",
                                       "action",
                                       "reward",
                                       "next_state",
                                       "terminal")
                        )


class BaseBuffer():
    """ Base class for the buffers. Push and sample
    methods need to be override. Initially start with
    an empty list(queue).

    Arguments:
        - capacity: Maximum size of the buffer
    """

    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity

    @property
    def size(self):
        """Return the current size of the buffer"""
        return len(self.queue)

    def __len__(self):
        """Return the capacity of the buffer"""
        return self.capacity

    def push(self, priority, *args, **kwargs):
        """Push transition into the buffer"""
        raise NotImplementedError

    def sample(self, batchsize, *args, **kwargs):
        """Sample transition from the buffer"""
        raise NotImplementedError



class PriorityBuffer(BaseBuffer):
    """ Replay buffer that sample tranisitons
    according to their prioirties. Prioirty
    value is most commonly td error.

    Arguments:
        - capacity: Maximum size of the buffer
        - min_prioirty: Values lower than the
        minimum prioirty value will be clipped
        - max_priority: Values larger than the
        maximum prioirty value will be clipped
    """

    def __init__(self, capacity, min_priority=0.1, max_priority=2):
        super().__init__(capacity)
        ### YOUR CODE HERE ###
        if capacity%2 != 0:
            raise ValueError('Please assign ''cepacity'' as product of 2' )
            self.capacity = capacity
        self.sumtree = SumTree(2 * capacity - 1) 
        self.min_priority = min_priority
        self.max_priority = max_priority
        self.alpha = 1 # [0~1] convert the importance of TD error to priority
        self.epsilon = 0.01         # small amount to avoid zero priority
        self.beta = 0.4             # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
#        raise NotImplementedError
        ###       END      ###

    def _clip_p(self, p):
        # You dont have to use this
        """ Return clipped priority """
        return min(max(p, self.min_priority), self.max_priority)

    def push(self, priority,**transition ):
        """ Push the transition with priority """
        ### YOUR CODE HERE ### 
#        if len(self.queue)%1000 == 0:
#            print(self.queue.__len__(),end='\r')
#        if len(self.queue)>4:
#            asd = randsample(self.queue, 4)
#            print(asd)
#            print(Transition(*zip(*asd)))
        if len(self.queue) == self.capacity:        # To make transition's index same with idnex of its priority
            self.queue[self.sumtree.data_pointer] = Transition(**transition)
        else:
            self.queue.append(Transition(**transition))
        priority += self.epsilon
        clipped_priority = priority
#        clipped_priority = self._clip_p(priority)
        self.sumtree.push(clipped_priority)
        
        #        raise NotImplementedError
        ###       END      ###

    def sample(self, batch_size):
        """ Return namedtuple of transition that
        is sampled with probability proportional to
        the priority values. """
        ### YOUR CODE HERE ###
#        raise NotImplementedError
        b_tree_idx, ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1))
        b_data = []
        total_priority = self.sumtree.tree[0]           # tree[0] is [root]partent maximum priority
        pri_seg = self.sumtree.tree[0] / float(batch_size) # priority segment
        counter = len(self.queue)
        if counter > self.capacity:
            min_prob = np.min(self.sumtree.tree[-self.capacity:]) / total_priority   
        else:
            min_prob = np.min(self.sumtree.tree[self.capacity-1:self.capacity-1+counter]) / total_priority # for later calculate ISweight
        for i in range(batch_size):
            
            a,b = pri_seg*i, pri_seg*(i+1)
            sel_pri     = np.random.uniform(a,b) # Selected Priority condition
            leaf_idx    = self.sumtree.get(sel_pri)
            data_idx    = leaf_idx - self.capacity+1
            priority    = self.sumtree.tree [leaf_idx]
            prob        = priority/total_priority            
            ISWeights[i, 0]     = np.power(prob/min_prob, -self.beta)            
            b_tree_idx[i]       = leaf_idx   
            b_data.append(self.queue[data_idx]) 
        b_data = Transition(*zip(*b_data))              # is added Lastly 
        return b_tree_idx, b_data, ISWeights
        ###       END      ###

    def update_priority(self, indexes, values):
        """ Update the prioirty value of the transition in
        the given index
        """
#        print(values)
        ### YOUR CODE HERE ###
#        raise NotImplementedError
        tree_idx = indexes # Tam olarak nasıl geldiğinden emin değilim kontrol et.
        values += self.epsilon
        for ti, p in zip(tree_idx, values):
#            print(p)
#            clipped_error = self._clip_p(p)
            clipped_error = p
            clipped_p = torch.pow(clipped_error, self.alpha)
            self.sumtree.update(ti, clipped_p)
        ###         END      ###
        
    def buffer_display(self,top_layers = None):
        leaf_num = self.capacity
#        print(leaf_num)
        k=0
        l=0
        if top_layers == None:
            top_layers = np.log2(self.capacity)+1
            
        for i  in range(self.sumtree.maxsize-1,-1,-1):
            l+=1
#            print(i)
            if k>np.log2(self.capacity)-top_layers:
                print(self.sumtree.tree[i],'   ',end='')
            if l%leaf_num==0:
                l=0
                k+=1
                leaf_num = round(leaf_num /2)
#                print(leaf_num)
                if k>np.log2(self.capacity)-top_layers:
                    print(end='\n')
                    print('      '*k,end='')
        print()
                
            
        
        
