
from Population import Population
from Algorithm import Algorithm
from time import time

start = time()

my_pop = Population(100, True)

generation_count = 0

while generation_count <= 1000:
        generation_count += 1
        print("Generation : %s Fittest : %s " % (generation_count, my_pop.get_elite_fitness()))
        my_pop = Algorithm.evolve_population(my_pop)
        print("******************************************************")
    


print("Solution found !\nGeneration : %s Fittest : %s " % (generation_count + 1, my_pop.get_elite_fitness()))


finish = time()
print ("Time elapsed : %s " % (finish - start)) 
