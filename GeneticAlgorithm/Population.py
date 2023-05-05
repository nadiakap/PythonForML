from Individual import Individual
from FitnessCalc import FitnessCalc

class Population():

    def __init__(self, population_size, initialise):
        self.individuals = []
        self.best_fitness = 0
        self.elite = 99

        #Creates the individuals
        if (initialise):
            for i in range(population_size):
                new_individual = Individual()
                self.individuals.append(new_individual)
            self.elite = self.get_fittest()
            self.best_fitness = self.fitness_of_the_fittest()
            
    def get_fitness(self, individual_passed):
        fitness = FitnessCalc.f(individual_passed)
        return fitness
     
    def fitness_of_the_fittest(self):
        f_fittest = FitnessCalc.f((self.individuals[0]).get_chromosome())
        for i in range(1,len(self.individuals)):
                f_ = FitnessCalc.f((self.individuals[i]).get_chromosome())
                if f_<f_fittest:
                    f_fittest=f_          
        return f_fittest            

    def get_elite(self):
        return self.elite
    
    def set_elite(self,individual):
         self.elite = individual
        
    def get_elite_fitness(self):
        return self.best_fitness
    
    def set_elite_fitness(self,fitness):
        self.best_fitness = fitness

    
    def get_fittest(self):
        fittest = self.individuals[0]
        f_fittest = FitnessCalc.f((self.individuals[0]).get_chromosome())
        for i in range(1,len(self.individuals)):
                f_ = FitnessCalc.f((self.individuals[i]).get_chromosome())
                if f_<f_fittest:
                    f_fittest=f_
                    fittest = self.individuals[i]
           
        return fittest

    def size(self):
        return len(self.individuals)
    
    def get_individual(self, index):
        return self.individuals[index]
        
    def save_individual(self, index, individual_passed):
        self.individuals[index] = individual_passed
