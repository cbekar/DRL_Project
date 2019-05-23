""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np


class SumTree():
    """ Binary heap with the property: parent node is the sum of
    two child nodes. Tree has a maximum size and whenever
    it reaches that, the oldest element will be overwritten
    (queue behaviour). All of the methods run in O(log(n)).

    Arguments
        - maxsize: Capacity of the SumTree

    """
    def __init__(self, maxsize):
        ### YOUR CODE HERE ###
        self.maxsize = maxsize
        self.data_pointer = 0
        self.tree = np.zeros(self.maxsize)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity  
        
#        self.data = np.zeros(self.maxsize, dtype=object)
        # [--------------data frame-------------]
		#             size: capacity

#        raise NotImplementedError
        ###       END      ###

    def push(self, priority):
        """ Add an element to the tree and with the given priority.
         If the tree is full, overwrite the oldest element.

        Arguments
            - priority: Corresponding priority value
        """
        ### YOUR CODE HERE ###
        tree_indx = int(self.data_pointer + (self.maxsize+1)/2-1 )
        self.update(tree_indx,priority)
        self.data_pointer += 1
        if self.data_pointer == (self.maxsize+1)/2:
            self.data_pointer = 0        
        
#        raise NotImplementedError
        ###       END      ###
        
    def get(self, priority):
        """ Return the node with the given priority value.
        Prioirty can be at max equal to the value of the root
        in the tree.

        Arguments
            - priority: Value whose corresponding index
                will be returned.
        """
        ### YOUR CODE HERE ###
        #        raise NotImplementedError
        
        parent_idx = 0
        while True:
            lc_idx = 2*parent_idx + 1       # Left Child index
            rc_idx = lc_idx + 1             # Right Child Index
            if lc_idx >= len(self.tree):    # Reached bottom, end search
                leaf_idx = parent_idx
                break
            else:                           # Continue Downward Search
                if priority <= self.tree[lc_idx]:# or self.tree[rc_idx] == 0.0:
                    parent_idx = lc_idx
                else:
                    priority -= self.tree[lc_idx]
                    parent_idx = rc_idx
        node = leaf_idx
        ###       END      ###
        return node

    def update(self, idx, value):
        """ Update the tree for the given idx with the
        given value. Values are updated via increasing
        the priorities of all the parents of the given
        idx by the difference between the value and
        current priority of that idx.

        Arguments
            - idx: Index of the data(not the tree).
            Corresponding index of the tree can be
            calculated via; idx + tree_size/2 - 1
            - value: Value for the node at pointed by
            the idx
        """
        ### YOUR CODE HERE ###
#        tree_indx = int(idx + (self.maxsize +1)/2 - 1)
        tree_indx = idx
        change    = value -self.tree[tree_indx]
        self.tree[tree_indx] = value
        while tree_indx != 0:
            tree_indx = (tree_indx-1)//2
            self.tree[tree_indx] +=change
            
#        raise NotImplementedError
        ###       END      ###
