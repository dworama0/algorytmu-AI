import random

from deap import base
from deap import creator
from deap import tools


creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax)

toolbox = base.Toolbox()

#Generator atrybutu
toolbox.register("attr_bools",random.randint,0,1)

#generator osobnika
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bools,100)

#generator populacji
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

#funkcja przystosowania
def evalOneMax(individual):
    return sum(individual),

#Ewaluacja
toolbox.register("evaluate",evalOneMax)

#opertor mutacji
toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)

#operator selekcji
toolbox.register("select",tools.selTournament,tournsize=3)

#operator krzyżowania
toolbox.register("mate", tools.cxTwoPoint)

def main():
    random.seed(64)
    #utworzenie populacji
    pop = toolbox.population(n=300)

    #CXPB - prawdopodobieństwo z który dwa osobniki zostaną reprodukowane
    #MUTPB - prawdopodobieństwo mutacji osobnika

    CXPB, MUTPB = 0.5,0.2

    print("Zaczynamy ewolucję....")
    print(len(pop))

    #ewaluacja populacji wejściowej
    fitnesses = list(map(toolbox.evaluate,pop))
    for ind,fit in zip(pop,fitnesses):
        ind.fitness.values = fit

    print(f" Ewaluacji poddano {len(pop)} osobników")

    #ekstrakcja ocen
    fits = [ind.fitness.values[0] for ind in pop]

    #generacja
    g=0

    #rozpoczęcie procesu ewolucji
    while max(fits) < 100 and g < 1000:
        g = g+1
        print(f"--- Generacja {g} ---")

        #selekcja
        offspring = toolbox.select(pop,len(pop))
        #klonowanie wybranych osobników
        offspring = list(map(toolbox.clone,offspring))

        #krzyżowanie i mutacja
        for child1,child2 in zip(offspring[::2],offspring[1::2]):

            #krzyżowanie
            if random.random() < CXPB:
                toolbox.mate(child1,child2)
                #susnięcie rodziców po procesie reprodukcji
                del child1.fitness.values
                del child2.fitness.values

            #mutacja
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values


        #ewaluacja osobników ze słabymi ocenami
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate,invalid_ind)
        for ind,fit in zip(invalid_ind,fitnesses):
            ind.fitness.values = fit

        print(f" --- Liczba ocenionych osobników: {len(invalid_ind)} --- ")

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits)/length
        sum2 = sum(x**2 for x in fits)
        std = abs(sum2/length - mean**2)**0.5

        print(f" --- Min -> {min(fits)}")
        print(f" --- Max -> {max(fits)}")
        print(f" --- Avg -> {mean}")
        print(f" --- Std -> {std}")
    print(" -- koniec ewolucji zakończony powodzeniem! --")

    best_ind = tools.selBest(pop,1)[0]
    print(f"-- Najepszy osobnik: {best_ind}: {best_ind.fitness.values}")

if __name__ == '__main__':
    main()
