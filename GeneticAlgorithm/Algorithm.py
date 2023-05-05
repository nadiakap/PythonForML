from Population import Population
from Individual import Individual
from Individual import generate_random_genes
from random import random
import numpy as np

class Algorithm():

   #Constants
   Uniform_rate = 0.5
   Mutation_rate = 0.15
   Tournament_size = 5
   Elitism_off_set = 1

   @staticmethod
   def evolve_population(population_passed):
      print("Evolving population...")
      new_population = Population(population_passed.size(), False)
      new_population.individuals.append(population_passed.get_elite())
      new_population.set_elite_fitness(population_passed.get_elite_fitness())
   
      #Do crossover over the entire population 
      for i in range(Algorithm.Elitism_off_set, population_passed.size()):
        individual1 = Algorithm.tournament_selection(population_passed)
        individual2 = Algorithm.tournament_selection(population_passed)
        new_individual = Algorithm.crossover(individual1, individual2)
        new_population.individuals.append(new_individual)
     
            
      #Do mutation randomly
      for i in range(Algorithm.Elitism_off_set, population_passed.size()):
         Algorithm.mutate(new_population.get_individual(i))
    
      new_population.set_elite(new_population.get_fittest())
      new_population.set_elite_fitness(new_population.fitness_of_the_fittest())
  
      return new_population
      
   @staticmethod
   def crossover(individual1_passed, individual2_passed):
      new_sol = Individual()
      for i in range(individual1_passed.size()):
         if random() <= Algorithm.Uniform_rate:
            new_sol.set_gene(i, individual1_passed.get_gene(i))
         else:
            new_sol.set_gene(i, individual2_passed.get_gene(i))
				
      return new_sol

   @staticmethod
   def mutate(individual_passed):
      for i in range(individual_passed.size()):
         if random() <= Algorithm.Mutation_rate:
            gene = generate_random_genes()
            individual_passed.set_gene(i, gene)

   @staticmethod
   def tournament_selection(population_passed):
      #Tournament pool
      tournament = Population(Algorithm.Tournament_size, False)

      """ Tournament selection technique.
		   How it works: The algorithm choose randomly five
		   individuals from the population and returns the fittest one """
      for i in range(Algorithm.Tournament_size):
         random_id = int(random() * population_passed.size())
         tournament.individuals.append(population_passed.get_individual(random_id))

      fittest = tournament.get_fittest()
      return fittest
