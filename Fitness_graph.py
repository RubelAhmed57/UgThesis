import numpy as np
import math
import  random as rn
import matplotlib.pyplot as plt
import pandas as pd
from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
import constants as Consts
from pyevolve import G1DList, GSimpleGA, Consts
import sys

#chromosom = [7,5,7,5,7,7,5,7,5,7]
means_men =[]
means_women =[]

def fitness(chromosom):
    chromosom =list(chromosom)
    #print chromosom

    fuzzy_div=chromosom[0:5]
    for i in range(len(fuzzy_div)):
        if(fuzzy_div[i]==0 or fuzzy_div[i]==1):
            fuzzy_div[i]=5

    #print fuzzy_div
    literal_dis=chromosom[5:10]
    for i in range (len(literal_dis)):
        literal_dis[i]=(literal_dis[i]-5.00)/30
    #print literal_dis
    reg = [1, 2, 3, 4, 5, 6, 7]
    # reg_x1 = [x+1 for x in range(fuzzy_div[0])]
    # reg_x2 = [x+1 for x in range(fuzzy_div[1])]
    # reg_x3 = [x+1 for x in range(fuzzy_div[2])]
    # reg_x4 = [x+1 for x in range(fuzzy_div[3])]
    # reg_y = [x+1 for x in range(fuzzy_div[4])]

    df = pd.read_csv('ele-2.csv')
    cols = len(df.columns)
    cols_header =  list(df)
    x1=  df['x1']
    x2 = df['x2']
    x3 = df['x3']
    x4 = df['x4']
    y = df['y']
    temp=[]

    #find region points ,fuzzy points finder

    def div_finder(ls, fuzzy_div , li_dis):
        points = []
        #print type(points)
        mx = np.float(max(ls))
        mn = np.float(min(ls))
        points.append(mn)
        nd = (mx - mn)/np.float(fuzzy_div-1)
        loop = fuzzy_div-1
        if(li_dis>0):
            for i in range(loop):
                if (i==0):
                    points.append(points[i] + nd)
                else:
                    points.append(points[i] + nd+ li_dis)
        else:
            for i in range(loop):
                if (i==loop):
                    points.append(points[i] + nd)
                else:
                    points.append(points[i] + nd+ li_dis)

        return points



    x1_points = div_finder(x1,fuzzy_div[0],literal_dis[0])
    x2_points = div_finder(x2,fuzzy_div[1],literal_dis[1])
    x3_points = div_finder(x3,fuzzy_div[2],literal_dis[2])
    x4_points = div_finder(x4,fuzzy_div[3],literal_dis[3])
    y_points  = div_finder(y,fuzzy_div[4], literal_dis[4])

    #print x1_points
    def div_a_b_c(points_list,li_dis):
        map_a_b_c=[[]]
        map_a_b_c[:] = []


        for i in range(len(points_list)):

            if(i==0):
                ls = [reg[i],points_list[i]-99,points_list[i],points_list[i+1]]
                map_a_b_c.append(ls)
            elif(i==(len(points_list)-1)):
                map_a_b_c.append([reg[i],points_list[i-1], points_list[i], points_list[i]+99]);
            else:
                map_a_b_c.append([reg[i],points_list[i-1], points_list[i], points_list[i+1]]);

        return map_a_b_c


    x1_a_b_c = div_a_b_c(x1_points,literal_dis[0])
    x2_a_b_c = div_a_b_c(x2_points,literal_dis[1])
    x3_a_b_c = div_a_b_c(x3_points,literal_dis[2])
    x4_a_b_c = div_a_b_c(x4_points,literal_dis[3])
    y_a_b_c  = div_a_b_c(y_points,literal_dis[4])

    #print  x1_a_b_c
    #print    x2_a_b_c


    # calculate degree:
    def mfDegree(point,aa,bb,cc):
        if point <aa:
            return 0.0
        elif aa<=point<bb:
            return (point-aa)/(bb-aa)
        elif bb<=point<cc:
            return (cc-point)/(cc-bb)
        else:
            return 0.0

    def range_finder(fuzzy_points,input_col,fuzzy_div): # finding associated regions
        range_index = []
        #points_len = fuzzy_div
        for i in range(len(input_col)):

            for j in range(fuzzy_div):
            # print x1[i]
                #print fuzzy_points[fuzzy_div-1]
                if((input_col[i]) >= (fuzzy_points[fuzzy_div-2])):
                   # print "entered" #calprit is here ...not entering any time.
                    #b,c = fuzzy_points[fuzzy_div-1]
                    range_index.append(fuzzy_div-2) # (fuzzy_div-1) also gives same resutl, ???
                    #a = fuzzy_points[fuzzy_div-2]
                    #c = fuzzy_points[fuzzy_div-1]
                    #b = fuzzy_points[fuzzy_div-1]
                    #print a,b,c
                    break
                elif(input_col[i] >= fuzzy_points[j] and input_col[i]< fuzzy_points[j+1]):
                    range_index.append(j)
                    #if(j == 0 ):
                        #a = fuzzy_points[j]
                    #else:
                       # a = fuzzy_points[j-1]
                    #b = fuzzy_points[j]
                    #c = fuzzy_points[j+1]
                    #print a,b,c*/

                #print x1[i]
        #print fuzzy_points[range_list]
        #late night edit
        #print fuzzy_points
        #for var in range(len(range_index)):
            #print range_index[var]

        asso_x = [[] for i in range(len(input_col))]
        #print asso_x

        for i in range(len(input_col)):
                asso_x[i].append(round(input_col[i],4))
                d1 = (input_col[i] - fuzzy_points[range_index[i]])/input_col[i]

                d2 = (fuzzy_points[range_index[i]+1] - input_col[i])/input_col[i]

                #print "\n"
                #print "d",d
                if(d1>d2):
                    d = range_index[i]+1
                    #print range_index[i]
                    asso_x[i].append(reg[d])
                    #print asso_x[i]
                    #print fuzzy_points[d]
                    #print "d",d
                else:
                    d = range_index[i]
                    #print range_index[i]
                    asso_x[i].append(reg[d])
                    #print asso_x[i]
                    #print fuzzy_points[d]
                    #print "d",d

                #print x1_reg[d]
                '''calculating a,b,c'''
                if(d==fuzzy_div-1):
                    c =  fuzzy_points[d]
                    b = fuzzy_points[d]
                    a =  fuzzy_points[d-1]
                elif(d==0):
                    a= fuzzy_points[d]
                    b =  fuzzy_points[d]
                    c =  fuzzy_points[1]
                else:
                    a= fuzzy_points[d-1]
                    b = fuzzy_points[d]
                    c = fuzzy_points[d+1]
                #print asso_x[i]

                #print input_col[i],a,b,c
                asso_x[i].append(mfDegree(input_col[i],a,b,c))

        #print asso_x   # associating input, universe, mfs
        return asso_x

    #print range_list


    x1_reg = []
    x1_reg= range_finder(x1_points,x1,fuzzy_div[0])

    x2_reg = []
    x2_reg= range_finder(x2_points,x2,fuzzy_div[1])

    x3_reg = []
    x3_reg= range_finder(x3_points,x3,fuzzy_div[2])

    x4_reg = []
    x4_reg= range_finder(x4_points,x4,fuzzy_div[3])
    y_reg = []

    y_reg= range_finder(y_points,y,fuzzy_div[4])

    #
    rules_list = [[] for i in range(len(x1))]
    refined_rules_list = [[]]
    #print len(refined_rules_list)

    for i in range(len(x1_reg)): #calculating rule degree
        #print x1_reg[i][2]*x2_reg[i][2]*x3_reg[i][2]*x4_reg[i][2]*y_reg[i][2]
         #print x1_reg[i][1],x2_reg[i][1],x3_reg[i][1],x4_reg[i][1],y_reg[i][1],x1_reg[i][2]*x2_reg[i][2]*x3_reg[i][2]*x4_reg[i][2]*y_reg[i][2]
         rules_list[i]=list((x1_reg[i][1],x2_reg[i][1],x3_reg[i][1],x4_reg[i][1],y_reg[i][1],x1_reg[i][2]*x2_reg[i][2]*x3_reg[i][2]*x4_reg[i][2]*y_reg[i][2],0))



    # for i in range (100):
    #     print rules_list[i]
    #writing all rule-base
    #f = open('workfil.dat', 'w')
    #for i in range(len(x1_reg)):
    #   f.write("IF X1 "+str(x1_reg[i])+", "+"X2 is "+ str(x2_reg[i]) + ", X3 is "+str(x3_reg[i])+ " AND X4 is "+str(x4_reg[i])+" THEN Y is "+str(y_reg[i])+"\n")
        #print x1_reg[2]*x2_reg[2]*x3_reg[2]*x1_reg[2]*y_reg[2]

    #sub_list = []

    #calculating refined rule list
    var = 0
    for i in range(len(rules_list)):
        if(rules_list[i][6] == 0):
            sub_list1 = rules_list[i][0:4]
            #print rules_list[i][0:5]
            #print "sub 1: ",sub_list1
            #print "printing sublist 1 ",  sub_list1
            rules_list[i][6] = 1
            mx_mul = rules_list[i][5]
            refined_point = i
            for j in range(len(rules_list)-i):
                sub_list2 = rules_list[j][0:4]
                #print "sub 2: ",sub_list2
                if(sub_list1 == sub_list2):
                    rules_list[j][6] = 1
                    #print "insert"
                    if(rules_list[j][5]>mx_mul):
                        mx_mul = rules_list[j][5]
                        refined_point = j
            var = var+1
            refined_rules_list.append(rules_list[refined_point])

    #print y_a_b_c

    #print var
    del(refined_rules_list[0])
    # for _ in range(len(refined_rules_list)):
    #
    #     print refined_rules_list[_]


    #sorted_refined_rules_list =  sorted(refined_rules_list,key=lambda x: (x[5]))

    #print refined_rules_list
    #test_data =[9.5,2.85,31.59,95.0]
    #print x1_a_b_c

    def MSE(test_data):
        sum_a=0.0001
        sum_a_x=0
        b=0

        #print test_data
        for i in range(50):
        #here i is every rule

            minDeg=1.0
            for j in range(5):
                if(j==0):

                    for k in range((fuzzy_div[0])-1):
                        #print x1_a_b_c[k][0],refined_rules_list[i][0]
                        if(x1_a_b_c[k][0]==refined_rules_list[i][0]):
                            #print "hello"
                            #print mfDegree(test_data[0],x1_a_b_c[1][k],x1_a_b_c[2][k],x1_a_b_c[3][k])
                            #print x1_a_b_c[k][1],x1_a_b_c[k][2],x1_a_b_c[k][3]
                            x=mfDegree(test_data[0],x1_a_b_c[k][1],x1_a_b_c[k][2],x1_a_b_c[k][3])
                            minDeg=min(x,minDeg)

                if(j==1):
                    for k in range((fuzzy_div[1])-1):
                        #print x2_a_b_c[k][0],refined_rules_list[i][1]
                        if(x2_a_b_c[k][0]==refined_rules_list[i][1]):
                            #print "hello 2"
                            #print mfDegree(test_data[1],x2_a_b_c[1][k],x2_a_b_c[2][k],x2_a_b_c[3][k])
                            x= mfDegree(test_data[1],x2_a_b_c[k][1],x2_a_b_c[k][2],x2_a_b_c[k][3])
                            minDeg=min(x,minDeg)

                if(j==2):
                    for k in range((fuzzy_div[2])-1):
                        #print x3_a_b_c[k][0],refined_rules_list[i][2]
                        if(x3_a_b_c[k][0]==refined_rules_list[i][2]):
                            #print "hello 3"
                            #print mfDegree(test_data[2],x3_a_b_c[1][k],x3_a_b_c[2][k],x3_a_b_c[3][k])
                            x= mfDegree(test_data[2],x3_a_b_c[k][1],x3_a_b_c[k][2],x3_a_b_c[k][3])
                            minDeg=min(x,minDeg)

                if(j==3):
                    for k in range((fuzzy_div[3])-1):
                        #print x4_a_b_c[k][0],refined_rules_list[i][3]
                        if(x4_a_b_c[k][0]==refined_rules_list[i][3]):
                            #print "hello 4"
                            #print mfDegree(test_data[3],x4_a_b_c[1][k],x4_a_b_c[2][k],x4_a_b_c[3][k])
                            x= mfDegree(test_data[3],x4_a_b_c[k][1],x4_a_b_c[k][2],x4_a_b_c[k][3])

                            minDeg=min(x,minDeg)


                if(j==4):
                    for k in range(fuzzy_div[4]-1):
                        #print y_a_b_c[k][0],refined_rules_list[i][4]
                        if(y_a_b_c[k][0]==refined_rules_list[i][4]):
                            #print "hello 5"
                            #print mfDegree(test_data[3],x4_a_b_c[1][k],x4_a_b_c[2][k],x4_a_b_c[3][k])
                            b = y_a_b_c[k][2]
                            #print b
            #print minDeg,b
            sum_a_x = sum_a_x+(minDeg*b)
            sum_a = sum_a+minDeg
        #print sum_a_x
        #print sum_a
        prediction= sum_a_x/sum_a
        actual=test_data[4]
        means_men.append(sum_a_x/sum_a + 0.01*test_data[4])
        means_women.append(test_data[4])
        return abs(((actual-prediction)/(actual))*100)
        #return abs((test_data[4]-(sum_a_x/sum_a)))
        #return  sum_a

    df1 = pd.read_csv('ele-3.csv')
    y=0
    z=0

#call the MSE function:

    for row in df1.iterrows():
        index, data = row
        #print data.tolist()

        x=MSE(data.tolist())
        y=y+x
        z=z+1
        #temp.append(data.tolist())

    print "\n"
    #print round(100-y/z,2),'%'

    print y/z
    print "Done"
    return z/y


#main function
#call of fitness function

#fitness(chromosom)



# Genome instance
genome = G1DList.G1DList(10)
genome.setParams(rangemin=3, rangemax=7)
#print genome

# The evaluator function (objective function)
genome.evaluator.set(fitness)
genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)

# Genetic Algorithm Instance

ga = GSimpleGA.GSimpleGA(genome)

#ga.getMinimax()
#mType1 = ga.getMinimax()

#Consts.minimaxType["minimize"]
#ga.setMinimax(Consts.minimaxType["minimize"])
#x=ga.getMinimax()


#ga.setMinimax(mType=mType1)
ga.selector.set(Selectors.GTournamentSelector)
ga.setPopulationSize(50)
ga.setGenerations(20)
ga.setElitism(True)
ga.setCrossoverRate(0.70)
ga.setMutationRate(0.10)


#best = ga.bestIndividual()
#min_score = best.score

# Do the evolution, with stats dump
# frequency of 10 generations
ga.evolve(freq_stats=1)

# Best individual
best = ga.bestIndividual()
#print type(ga.bestIndividual())

#plot code
means_men = means_men[-60:]
means_women = means_women[-60:]
n_groups = len(means_men)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.70
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='predicted')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Actual')

plt.xlabel('Samples')
plt.ylabel('Values')
plt.title('')
#plt.xticks(np.arange(min(x), max(x)+1, 5.0))
#plt.xticks(index + bar_width / 2, index)
plt.xticks(np.arange(0,n_groups, 5.0))
plt.legend()

plt.tight_layout()

#y=0
#z=0
# df1 = pd.read_csv('ele-3.csv')
# for row in df1.iterrows():
#     index, data = row
#     #print data.tolist()
#     x=MOD_MSE(data.tolist())
#     y=y+x
#     z=z+1
#
# print y/z
plt.show()



