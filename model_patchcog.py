# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:42:23 2019

@author: mahi
"""

from mesa import Agent, Model
from mesa.space import SingleGrid
import random
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
#import matplotlib.animation as plt
import numpy as np
from mesa.space import Grid
from mesa.datacollection import DataCollector
import scipy.stats as st
from matplotlib import colors
from mesa.time import BaseScheduler
import math
import pandas as pd
import sys

# add colors to graphics
# add step number to graphics

class A2(Agent):  # second level agent-- predator agent   
    
    """ animal agent functions
    """
    
    def __init__(self, unique_id, model, start_energy, cognition, disp_rate):
        super().__init__(unique_id, model)  # creates an agent in the world with a unique id
        self.energy = self.model.start_energy
        self.eat_energy = self.model.eat_energy
        self.tire_energy = self.model.tire_energy
        self.reproduction_energy = self.model.reproduction_energy
        if self.model.cog_fixed:
            self.cognition = (self.model.dist, self.model.det)
        else:
            self.cognition = cognition
        self.cognition_energy = self.model.cognition_energy
        self.dead = False
        self.identity = "A2"
        self.age = 0
        if self.model.evolve_disp == True:
            self.disp_rate = disp_rate
            disp = np.power(self.disp_rate, range(0,100))
            self.disp = disp/sum(disp)
        else:
            self.disp_rate = self.model.disp_rate
            self.disp = self.model.disp
                
    def step(self): # this iterates at every step
        if not self.dead:  # the agent moves on ev ery step   
            self.cognition_and_move()
        if not self.dead:
            self.tire_die()
        if not self.dead:
            self.reproduce()
            self.age+=1
            self.model.age.append(self.age)
            
    def introduce(self, x, y, energy, cog, disp):
        a = A2(self.model.unique_id, self.model, start_energy = energy, cognition = cog, disp_rate = disp)
        self.model.unique_id += 1
        self.model.grid.place_agent(a, (x,y))
        self.model.schedule.add(a)
        self.model.agentgrid[x][y] = 2
        self.model.coggrid[:, x, y] = a.cognition
        self.model.dispgrid[1,x,y] = a.disp_rate
        
    def kill(self):
        self.dead=True
        x,y = self.pos
        self.model.grid.remove_agent(self) 
        self.model.schedule.remove(self)
        self.model.agentgrid[x][y] = 0
        self.model.coggrid[:, x, y] = tuple([101] * 2)#self.model.nCogPar)
        self.model.dispgrid[1, x, y] = 101
        self.model.death += 1
        
    def move(self, coord):
        x,y = self.pos
        newx, newy = coord
        self.model.grid.move_agent(self, coord)
        self.model.agentgrid[newx][newy] = 2
        self.model.coggrid[:, newx, newy] = self.cognition
        self.model.dispgrid[1, newx, newy ] = self.disp_rate
        self.model.agentgrid[x][y] = 0 
        self.model.coggrid[:, x, y] = tuple([101] * 2) #self.model.nCogPar)
        self.model.dispgrid[1, x, y] = 101
        
    def eat(self, coord, eat_now = True):
        die = self.model.grid.get_cell_list_contents([coord])[0]
        food = die.energy
        self.model.food += food
        die.dead = True
        self.model.grid.remove_agent(die)
        self.model.schedule.remove(die)   
        x,y = coord
        self.model.agentgrid[x][y]= 0
        self.model.dispgrid[0,x,y]=101
        if eat_now:
            self.energy += food
        else:
            return(food)
      
    def reproduce(self): # reproduce function
        if self.energy >= self.reproduction_energy:
            self.model.reprod += 1
            if self.model.disp_rate == 1:
                x = random.randrange(self.model.grid.width)
                y = random.randrange(self.model.grid.height)
                new_position = (x,y)
            elif self.disp_rate == 0:
                possible_locs = self.model.grid.get_neighborhood(   # position of new ofspring
                self.pos,
                moore=True,
                include_center=False)
                new_position = random.choice(possible_locs)
            else:
                xnum = abs(100-self.pos[0])
                if self.pos[0]>100:
                    xnum = 200-xnum
                ynum = abs(100-self.pos[1])
                if self.pos[1]>100:
                    ynum = 200-ynum

                p = self.model.positions
                p = np.concatenate((p[xnum:, :], p[:xnum, :]))
                positions = np.concatenate((p[:, ynum:], p[:, :ynum]), 1)
                dist = random.choices( range(100), k=1, weights = self.disp )[0]
                new_position = tuple(random.choice(list(np.argwhere( positions == (dist+1) )) ))
                
            energy_own = math.ceil(self.energy/2)
            energy_off = self.energy - energy_own
            self.energy = energy_own
            
            cog = self.cognition
            
            if not self.model.cog_fixed: # if model is 0 cogtype, don't evolve
                p = random.choice([0,1])
                random_ = np.random.normal(0,0.05,1)[0]
                new = max(min(cog[p] + random_, 1), 0)  #max value goes in the inner bracket, min goes in the outr bracket#                            
                cog = ( *cog[0:p], new, *cog[p+1:])
                        
            new_disp = self.disp_rate
            if self.model.evolve_disp == True:
                random_ = np.random.normal(0,0.025,1)[0]
                new_disp = max(min(self.disp_rate + random_, 1), 0)

            x,y = new_position                
            if self.model.agentgrid[x][y] == 1:
                food = self.eat((x,y), eat_now = False)
                self.introduce(x,y, energy_off + food, cog, new_disp)
                
            elif self.model.agentgrid[x][y] == 0:
                self.introduce(x,y,energy_off, cog, new_disp)
            
    def tire_die(self): 
        x,y = self.pos
        self.energy-=self.tire_energy # + (self.cognition[0]/10)
        if self.energy<=0:
            self.kill()
                    
    def cogdecision(self):
        neighbors = self.model.grid.get_neighborhood(
                    self.pos,
                    moore=True,
                    include_center=False)
        
        if (self.cognition[1]==0):
            new_pos = random.choice(neighbors)
        else:
            (a1weights, a2weights) = (np.array([]), np.array([]))

            weight = self.cognition[0]
            for n in neighbors:                
                weights__ = np.power(weight, self.model.positions_food)
                food__ = self.toroidal( (n[0]-10)%200, (n[0]+10+1)%200, (n[1]-10)%200, (n[1]+10+1)%200 )
                a1weights = np.append(a1weights, np.sum(weights__*food__))
            
            a2weights = np.array([0]*8)
            
            (a1wt_, a2wt_, k ) = (self.cognition[1], 0, 4)
            
            a1wt = a1wt_ * a1weights
            a2wt = a2wt_ * a2weights
            wt = a1wt + a2wt
            wtexp = np.exp(wt*k)
            
            inf_check = np.argwhere(np.isinf(wtexp))
            if len(inf_check)==1:
                idx = int(inf_check[0])
                print(idx)
                return(neighbors[idx])
            if len(inf_check)>1:
                wtexp = np.exp(wt*1.6)
            
            wtfinal = wtexp/np.sum(wtexp)
            new_pos = random.choices( neighbors, k=1, weights = wtfinal )[0]
    
        return (new_pos)
                
    
    def cognition_and_move(self):  
        self.energy-=self.cognition_energy  
        new_pos = self.cogdecision()
        newx, newy = new_pos
        x,y = self.pos
        if self.model.agentgrid[newx][newy] == 1:
            self.eat(new_pos)
            self.move(new_pos)
        elif self.model.agentgrid[newx][newy] == 0:
            self.move(new_pos)
        elif self.model.agentgrid[newx][newy] >= 2:
            self.model.combat += 1
            combat = self.model.grid.get_cell_list_contents([new_pos])[0]
            coin = random.random()
            if combat.energy>self.energy or (combat.energy==self.energy and coin<0.5):
                self.kill()
            else:
                combat.kill()
                self.move(new_pos)

    def toroidal(self, start1, end1, start2, end2, gridsize=200):
        array = self.model.agentgrid
        if start1<end1 and start2<end2:
            array = array[start1:end1, start2:end2]
        elif start1>end1 and start2<end2:
            array1 = array[start1:gridsize, start2: end2]
            array2 = array[0:end1, start2: end2]
            array = np.concatenate((array1, array2))
        elif start1<end1 and start2>end2:
            array1 = array[start1:end1, start2:gridsize]
            array2 = array[start1:end1, 0:end2]
            array  = np.concatenate((array1, array2), 1 )
        else:
            array1 = array[start1:gridsize, start2:gridsize]
            array2 = array[start1:gridsize, 0:end2]
            array3 = array[0:end1, start2:gridsize]
            array4 = array[0:end1, 0:end2]
            array = np.concatenate(( np.concatenate((array1, array2), 1 ), \
                                   np.concatenate((array3, array4), 1 )))
        return(array == 1)

class A1(Agent):
    
    """ plants agent functions
    """
    
    def __init__(self, unique_id, model, energy, disp_rate):
        super().__init__(unique_id, model)
        self.energy = self.model.start_energy # agent starts at energy level 10
        self.eat_energy =  self.model.eat_energy
        self.tire_energy = self.model.tire_energy
        self.reproduction_energy = self.model.reproduction_energy
        self.dead = False
        self.identity = "A1"
        if self.model.evolve_disp == True:
            self.disp_rate = disp_rate
            disp = np.power(self.disp_rate, range(0,100))
            self.disp = disp/sum(disp)
        else:
            self.disp_rate = self.model.disp_rate
            self.disp = self.model.disp
        
    def step(self): # this iterates at every step
        self.eat()
        self.tire_die()
        if not self.dead:
            self.reproduce()
            
    def reproduce(self):
        if self.energy>=self.reproduction_energy:
            if self.disp_rate == 1:
                x = random.randrange(self.model.grid.width)
                y = random.randrange(self.model.grid.height)
                new_position = (x,y)
            elif self.disp_rate == 0:
                possible_locs = self.model.grid.get_neighborhood(   # position of new ofspring
                self.pos,
                moore=True,
                include_center=False)
                new_position = random.choice(possible_locs)
            else:
                xnum = abs(100-self.pos[0])
                if self.pos[0]>100:
                    xnum = 200-xnum
                ynum = abs(100-self.pos[1])
                if self.pos[1]>100:
                    ynum = 200-ynum

                p = self.model.positions
                p = np.concatenate((p[xnum:, :], p[:xnum, :]))
                positions = np.concatenate((p[:, ynum:], p[:, :ynum]), 1)
                dist = random.choices( range(100), k=1, weights = self.disp )[0]
                new_position = tuple(random.sample(list(np.argwhere( positions == (dist+1) )), k=1)[0])
            
            energy_own = math.ceil(self.energy/2)
            energy_off = self.energy - energy_own
            self.energy = energy_own
            
            new_disp = self.disp_rate
            if self.model.evolve_disp == True:  
                random_ = np.random.normal(0,0.025,1)[0]
                new_disp = max(min(self.disp_rate + random_, 1), 0)
            
            if ( self.model.grid.is_cell_empty(new_position)):
                a = A1(self.model.unique_id, self.model, energy_off, new_disp)
                self.model.unique_id += 1
                self.model.grid.place_agent(a, new_position)
                self.model.schedule.add(a)
                x,y = new_position
                self.model.agentgrid[x][y] = 1
                self.model.dispgrid[0,x,y] = new_disp
                

    def eat(self): # agent eats at every step and thus depeletes resources          
        self.energy += self.eat_energy # nutrition is added to agent's nutrition
            
    def tire_die(self): # agent loses energy at every step. if it fails to eat regularly, it dies due to energy loss
        x,y = self.pos
        self.energy-=self.tire_energy
        if self.energy<=0:
            self.dead=True
            self.model.grid[x][y].remove(self) 
            self.model.schedule.remove(self)
            self.model.agentgrid[x][y] -= 1 
            self.model.dispgrid[0,x,y]  =101


class modelSim(Model):

    """ 
    details of the world 
    
    introduce time is when animal agents first get introduced into the wrold
    disp_rate is the dispersal rate for experiment 3
    dist is perceptual strength for animals if fixed
    det is decision determinacy of animals if fixed
    cog_fixed determines if cognition of animals is fixed to particular values or is allowed to evolve
    if skip_300 is True, patchiness values are not calculated for the first 300 steps-- this makes the model run faster
    collect_cog_dist creates a seperate dataframe for all cognition values for agents at every timestep
    if evolve_disp is true, dispersion rate of plants is free to evolve
    """
    
    def __init__(self, introduce_time, disp_rate, dist, det, cog_fixed = False, \
                 skip_300 = True, collect_cog_dist = False, evolve_disp = False):
        
        
        self.skip_300 = skip_300
        self.cog_fixed = cog_fixed
        self.evolve_disp = evolve_disp
        self.collect_cog_dist = collect_cog_dist
        self.dist = dist
        self.det = det
        self.disp_rate = disp_rate
        self.intro_time = introduce_time
        (self.a1num, self.a2num) = (20, 20)
        self.schedule = RandomActivation(self) # agents take a step in random order 
        self.grid = SingleGrid(200, 200, True) # the world is a grid with specified height and width
        
        self.initialize_perception()
            
        disp = np.power(self.disp_rate, range(0,100))
        self.disp = disp/sum(disp)
        self.grid_ind = np.indices((200,200))
        positions = np.maximum(abs(100-self.grid_ind[0]), 
                               abs(100-self.grid_ind[1]) )
        self.positions = np.minimum(positions, 200-positions)
        
        self.agentgrid = np.zeros((self.grid.width, self.grid.height)) # allows for calculation of patchiness of both agents
        self.coggrid = np.full((self.nCogPar, self.grid.width, self.grid.height), 101.0)
        self.dispgrid = np.full((2, self.grid.width, self.grid.height), 101.0 )                         
        self.age = []
        (self.nstep, self.unique_id, self.reprod, self.food, self.death, self.combat) = (0, 0, 0, 0, 0, 0)
        
        self.cmap = colors.ListedColormap(['midnightblue', 'mediumseagreen', 'white', 'white', 'white', 'white', 'white'])#'yellow', 'orange', 'red', 'brown'])
        bounds=[0,1,2,3,4,5,6,7]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)    
        
        self.expect_NN = []
        self.NN = [5, 10]
        for i in self.NN:
            self.expect_NN.append((math.factorial(2*i) * i)/(2**i * math.factorial(i))**2)
            
        grid_ind_food = np.indices((21,21))
        positions_food = np.maximum(abs(10-grid_ind_food[0]), abs(10-grid_ind_food[1]) )
        self.positions_food = np.minimum(positions_food, 21 - positions_food)
        if self.collect_cog_dist:
            self.cog_dist_dist = pd.DataFrame(columns = [])
            self.cog_dist_det = pd.DataFrame(columns = [])

        for i in range(self.a1num): # initiate a1 agents at random locations
            self.introduce_agents("A1")
        self.nA1 = self.a1num
        self.nA2 = 0
   #     self.agent_steps = {}

    def initialize_perception(self):
        self.history = pd.DataFrame(columns = ["nA1", "nA2", "age", "LIP5", "LIP10", "LIPanim5", "LIPanim10", "Morsita5", "Morsita10", "Morsitaanim5", "Morsitaanim10", "NN5","NN10","NNanim5", "NNanim10", "reprod", "food", "death", 
                                       "combat", "dist", "det", "dist_lower", "det_lower", "dist_upper", "det_upper", "dist_ci", "det_ci"])
        self.nCogPar = 2
        (self.start_energy, self.eat_energy, self.tire_energy, self.reproduction_energy, self.cognition_energy) \
        = (10, 5, 3, 20, 1)

            
    def introduce_agents(self, which_agent):
        x = random.randrange(self.grid.width)
        y = random.randrange(self.grid.height)
            
        if which_agent == "A1":
            if self.grid.is_cell_empty((x,y)):
                a = A1(self.unique_id, self, self.start_energy, disp_rate = 0)
                self.unique_id += 1
                self.grid.position_agent(a, x, y)
                self.schedule.add(a)
                self.agentgrid[x][y] = 1
            else:
                self.introduce_agents(which_agent)
        elif which_agent == "A2":
            if self.cog_fixed:
                c = (self.dist, self.det)
            else:
                c = tuple([0]*self.nCogPar)
            a = A2(self.unique_id, self, self.start_energy, cognition = c, disp_rate = 0)
            self.unique_id += 1
            if self.agentgrid[x][y] == 1:
                die = self.grid.get_cell_list_contents([(x,y)])[0]
                die.dead = True
                self.grid.remove_agent(die)
                self.schedule.remove(die)
                self.grid.place_agent(a, (x,y))
                self.schedule.add(a)
                self.agentgrid[x][y] = 2
                self.coggrid[:, x, y] = c
            elif self.agentgrid[x][y] == 0:
                self.grid.place_agent(a, (x,y))
                self.schedule.add(a)    
                self.agentgrid[x][y] = 2
                self.coggrid[:, x, y] = c
    
    def flatten_(self, n, grid, full_grid = False, mean = True, range_ = False):
        if full_grid:
            return(grid[n].flatten())
        i = grid[n].flatten()
        if mean:
            i = np.delete(i, np.where(i == 101))
            if len(i) == 0:
           # if range_:
               return([0]*4)
            #else:
            #    return(0)
            if range_:
                if self.cog_fixed:
                    return([np.mean(i)]*4)
                return( np.concatenate( ( [np.mean(i)], np.percentile(i, [2.5, 97.5]), self.calculate_ci(i) )) )
            return([np.mean(i), 0, 0, 0])
        else:
            return(i)
    
    def calculate_ci(self, data):
        if np.min(data) ==np.max(data):
            return( [ 0.0])
        return ( [np.mean(data) - st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]])
    
    def return_zero(self, num, denom):
        if self.nstep == 1:
       #     print("whaaat")
            return(0)
        if denom == "old_nA2":
            denom = self.history["nA2"][self.nstep-2]
        if denom == 0.0:
            return 0
        return(num/denom)
        
    def nearest_neighbor(self, agent):     # fix this later
        if agent == "a1":
            x = np.argwhere(self.agentgrid==1)
            if len(x)<=10:
                return([-1]*len(self.NN))
            elif len(x) > 39990:
                return([0.97, 0.99])
          #  if self.nstep<300 and self.skip_300:
          #      return([-1,-1] )
        else:
            x = np.argwhere(self.agentgrid==2)
            if len(x)<=10:
                return([-1]*len(self.NN))
        density = len(x)/ (self.grid.width)**2
        expect_NN_ = self.expect_NN
        expect_dist = np.array(expect_NN_) /(density ** 0.5)
        distances = [0, 0]
        for i in x:
            distx = abs(x[:,0]-i[0])
            distx[distx>100] = 200-distx[distx>100]
            disty = abs(x[:,1]-i[1])
            disty[disty>100] = 200-disty[disty>100]
            dist = (distx**2+disty**2)**0.5
            distances[0] += (np.partition(dist, 5)[5])
            distances[1] += (np.partition(dist, 10)[10])
        mean_dist = np.array(distances)/len(x)
        out = mean_dist/expect_dist
        return(out)
   
    def quadrant_patch(self, agent):  # function to calculate the patchiness index of agents at every step
        if agent == "a1":
            x = self.agentgrid == 1
        else:
            x = self.agentgrid == 2
        gsize = np.array([5,10])
        gnum = 200/gsize
        qcs = []
        for i in range(2):
            x_ = x.reshape(int(gnum[i]), gsize[i], int(gnum[i]), gsize[i]).sum(1).sum(2)
            mean = np.mean(x_)
            var = np.var(x_)
            if mean==0.0:
                return([-1]*4)
            lip = 1 + (var-mean) / (mean**2)
            morsita = np.sum(x) * ( (np.sum(np.power(x_, 2)) - np.sum(x_))/( np.sum(x_)**2 - np.sum(x_)))
            qcs += [lip, morsita]
        return(qcs)
        
    def l_function(self, agent):
        if agent == "a1":
            x = np.argwhere(self.agentgrid==1)
        else:
            x = np.argwhere(self.agentgrid==2)
            if len(x)==0:
                return(-1)
        distances = np.array([])
        for i in x:
            distx = abs(x[:,0]-i[0])
            distx[distx>100] = 200-distx[distx>100]
            disty = abs(x[:,1]-i[1])
            disty[disty>100] = 200-disty[disty>100]
            dist = (distx**2 + disty**2)**0.5
            distances = np.concatenate((distances, dist[dist!=0]))
        l = np.array([])
        for i in np.arange(5, 51, 5):
            l = np.append(l, sum(distances<i))
        k = (l * 200**2) / (len(x)**2)
        l = (k/math.pi)**0.5         
        return(abs(l - np.arange(5, 51, 5)))
    
    def collect_hist(self):
        if self.nstep<300 and self.skip_300:
            NNcalc = [-1, -1]#self.nearest_neighbor("a1") 
            NNanimcalc = [-1, -1]#self.nearest_neighbor("a2")
        else:
            NNcalc = self.nearest_neighbor("a1") 
            NNanimcalc = self.nearest_neighbor("a2")
        quadrantcalc = self.quadrant_patch( "a1")
        quadrantanimcalc = self.quadrant_patch( "a2")
        dist_values = self.flatten_(0, grid = self.coggrid, mean = True, range_ = False)
        det_values = self.flatten_(1,  grid = self.coggrid, mean = True, range_ = False)
       # l_f = 0#self.l_function("a1")
        dat = { "nA1" : self.nA1, "nA2" : self.nA2,
               "age" : self.return_zero(sum(self.age), self.nA2),
               "LIP5" :   quadrantcalc[0], "LIP10" :   quadrantcalc[2],
               "LIPanim5":  quadrantanimcalc[0], "LIPanim10":  quadrantanimcalc[2],
               "Morsita5" :   quadrantcalc[1], "Morsita10" :   quadrantcalc[3],
               "Morsitaanim5": quadrantanimcalc[1], "Morsitaanim10": quadrantanimcalc[3],
               "NN5": NNcalc[0],"NN10": NNcalc[1],
               "NNanim5": NNanimcalc[0],"NNanim10": NNanimcalc[1], #"l_ripley" : l_f,# self.nearest_neighbor("a2"),  
               "reprod" : self.return_zero(self.reprod, "old_nA2" ), "food": self.return_zero(self.food, self.nA2),
               "death" : self.return_zero(self.death, "old_nA2"), "combat" : self.return_zero(self.combat, "old_nA2"),
               "dist" : dist_values[0], "det" : det_values[0],
               "dist_lower" : dist_values[1], "det_lower" : det_values[1],
               "dist_upper" : dist_values[2], "det_upper" : det_values[2],
               "dist_ci" : dist_values[3], "det_ci" : det_values[3], 
               "disp_a1" : self.flatten_(0, grid = self.dispgrid)[0], "disp_a2" : self.flatten_(1, grid = self.dispgrid)[0] }
        self.history = self.history.append(dat, ignore_index = True)
        self.age = []
        (self.reprod, self.food, self.death, self.combat) = (0, 0, 0, 0)
        if self.collect_cog_dist:
            if (self.nstep %10) == 0:
                self.cog_dist_dist[str(self.nstep-1)] = self.flatten_(0, grid = self.coggrid, full_grid = True, mean=False)
                self.cog_dist_det[str(self.nstep-1)] = self.flatten_(1, grid = self.coggrid, full_grid = True, mean=False)
  
    def step(self):
        self.nstep +=1 # step counter
        if self.nstep == self.intro_time:
            for i in range(self.a2num):
                self.introduce_agents("A2")  
        self.schedule.step()  
        self.nA1 = np.sum(self.agentgrid==1)            
        self.nA2 = np.sum(self.agentgrid==2)
        self.collect_hist()
        if self.nstep%10 == 0:
            sys.stdout.write( (str(self.nstep) +" "  +str(self.nA1) + " " + str(self.nA2) + "\n") )
        
    def visualize(self):
        f, ax = plt.subplots(1)
        self.agentgrid = self.agentgrid.astype(int)
        ax.imshow(self.agentgrid, interpolation='nearest', cmap=self.cmap, norm=self.norm)
       # plt.axis("off")
        return(f)
