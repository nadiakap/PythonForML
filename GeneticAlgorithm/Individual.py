import numpy as np
DefaultGeneLength = 2

    
def generate_random_genes(sz = None):
    return np.random.uniform(-5,5,size=(sz))
                             
class Individual():
  

   def __init__(self):
      #self.genes =(np.random.uniform(-5,5,size=(DefaultGeneLength)))
      self.genes =generate_random_genes(DefaultGeneLength)

   def get_chromosome(self):
	    return self.genes
    
   def get_gene(self, index):
	   return self.genes[index]
	
   def set_gene(self, index, what_to_set):
	    self.genes[index] =  what_to_set
	
   def size(self):
	    return len(self.genes)
