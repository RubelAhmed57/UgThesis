import pandas as pd
import numpy as np
from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators

reg = ['S2', 'S1', 'CE', 'B1', 'B2']
df = pd.read_csv('ele-2.csv')
cols = len(df.columns)
cols_header =  list(df)
x1=  df['x1']
x2 = df['x2']
x3 = df['x3']
x4 = df['x4']
y = df['y']
temp=[]


#print x1[0] + float(1.000)
#print "printing type of x1[0] : ",type(x1[0])
#x1 = x1.sort()

fuzzy_div = 5

#find region points ,fuzzy points finder
def div_finder(ls):
    points = []
    #print type(points)
    mx = np.float(max(ls))
    mn = np.float(min(ls))
    points.append(mn)
    nd = (mx - mn)/np.float(fuzzy_div-1)
    #print mx, mn,nd
    #print "print type of nd: ", type(nd)
    #print nd  #show rajesh
    loop = fuzzy_div-1
    for i in range(loop):
         #print "printing points[i]",points[i]
         points.append(points[i] + nd)
         #print points[i+1]
    #print "printing list: ", points
    return points



x1_points = []
x2_points = []
x3_points = []
x4_points = []
y_points = []

x1_points = div_finder(x1)
x2_points = div_finder(x2)
x3_points = div_finder(x3)
x4_points = div_finder(x4)
y_points  = div_finder(y)


def div_a_b_c(points_list):
    map_a_b_c=[[]]
    map_a_b_c[:] = []


    for i in range(len(points_list)):

        if(i==0):
            ls = [reg[i],points_list[i],points_list[i],points_list[i+1]]
            map_a_b_c.append(ls)
        elif(i==4):
            map_a_b_c.append([reg[i],points_list[i-1],points_list[i],points_list[i]]);
        else:
            map_a_b_c.append([reg[i],points_list[i-1],points_list[i],points_list[i+1]]);
    return map_a_b_c


x1_a_b_c=(div_a_b_c(x1_points))
x2_a_b_c=(div_a_b_c(x2_points))
x3_a_b_c=(div_a_b_c(x3_points))
x4_a_b_c=(div_a_b_c(x4_points))
y_a_b_c=(div_a_b_c(y_points))

#print x1_a_b_c
#print x1_points
#print x1_points
#print len(x1_points)

#print x1_n_d 

#print x2_points



# region names
#print x1_reg[1]

# finding a,b,c for x1

# x1_abc = []


#check both region
#x1_reg= range_finder(x1_points,x1)
#abc will be found here
#print x1_points[4],type(x1_points[4]) #show rajesh

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

def range_finder(fuzzy_points,input_col): # finding associated regions
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
            if(d==4):
                c =  fuzzy_points[d]
                b = fuzzy_points[d]
                a =  fuzzy_points[3]
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




          
    #print asso_x
    return asso_x 

#print range_list
            
            
x1_reg = []
x1_reg= range_finder(x1_points,x1)
 
x2_reg = []
x2_reg= range_finder(x2_points,x2)

x3_reg = []
x3_reg= range_finder(x3_points,x3)

x4_reg = []
x4_reg= range_finder(x4_points,x4)
y_reg = []

y_reg= range_finder(y_points,y)

#
rules_list = [[] for i in range(len(x1))]
refined_rules_list = [[]]
#print len(refined_rules_list)

for i in range(len(x1_reg)): #calculating rule degree
    #print x1_reg[i][2]*x2_reg[i][2]*x3_reg[i][2]*x4_reg[i][2]*y_reg[i][2]
     #print x1_reg[i][1],x2_reg[i][1],x3_reg[i][1],x4_reg[i][1],y_reg[i][1],x1_reg[i][2]*x2_reg[i][2]*x3_reg[i][2]*x4_reg[i][2]*y_reg[i][2]
     rules_list[i]=list((x1_reg[i][1],x2_reg[i][1],x3_reg[i][1],x4_reg[i][1],y_reg[i][1],x1_reg[i][2]*x2_reg[i][2]*x3_reg[i][2]*x4_reg[i][2]*y_reg[i][2],0))




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
#for _ in range(len(refined_rules_list)):

    #print refined_rules_list[_]



#test_data =[9.5,2.85,31.59,95.0]
#print x1_a_b_c

def MSE(test_data):
    sum_a=0
    sum_a_x=0
    b=0
    #print test_data
    for i in range(50):
    #here i is every rule

        minDeg=1.0
        for j in range(5):
            if(j==0):

                for k in range(4):
                    #print x1_a_b_c[k][0],refined_rules_list[i][0]
                    if(x1_a_b_c[k][0]==refined_rules_list[i][0]):
                        #print "hello"
                        #print mfDegree(test_data[0],x1_a_b_c[1][k],x1_a_b_c[2][k],x1_a_b_c[3][k])
                        #print x1_a_b_c[k][1],x1_a_b_c[k][2],x1_a_b_c[k][3]
                        x=mfDegree(test_data[0],x1_a_b_c[k][1],x1_a_b_c[k][2],x1_a_b_c[k][3])
                        minDeg=min(x,minDeg)

            if(j==1):
                for k in range(4):
                    #print x2_a_b_c[k][0],refined_rules_list[i][1]
                    if(x2_a_b_c[k][0]==refined_rules_list[i][1]):
                        #print "hello 2"
                        #print mfDegree(test_data[1],x2_a_b_c[1][k],x2_a_b_c[2][k],x2_a_b_c[3][k])
                        x= mfDegree(test_data[1],x2_a_b_c[k][1],x2_a_b_c[k][2],x2_a_b_c[k][3])
                        minDeg=min(x,minDeg)

            if(j==2):
                for k in range(4):
                    #print x3_a_b_c[k][0],refined_rules_list[i][2]
                    if(x3_a_b_c[k][0]==refined_rules_list[i][2]):
                        #print "hello 3"
                        #print mfDegree(test_data[2],x3_a_b_c[1][k],x3_a_b_c[2][k],x3_a_b_c[3][k])
                        x= mfDegree(test_data[2],x3_a_b_c[k][1],x3_a_b_c[k][2],x3_a_b_c[k][3])
                        minDeg=min(x,minDeg)

            if(j==3):
                for k in range(4):
                    #print x4_a_b_c[k][0],refined_rules_list[i][3]
                    if(x4_a_b_c[k][0]==refined_rules_list[i][3]):
                        #print "hello 4"
                        #print mfDegree(test_data[3],x4_a_b_c[1][k],x4_a_b_c[2][k],x4_a_b_c[3][k])
                        x= mfDegree(test_data[3],x4_a_b_c[k][1],x4_a_b_c[k][2],x4_a_b_c[k][3])

                        minDeg=min(x,minDeg)


            if(j==4):
                for k in range(4):
                    #print y_a_b_c[k][0],refined_rules_list[i][4]
                    if(y_a_b_c[k][0]==refined_rules_list[i][4]):
                        #print "hello 5"
                        #print mfDegree(test_data[3],x4_a_b_c[1][k],x4_a_b_c[2][k],x4_a_b_c[3][k])
                        b= y_a_b_c[k][2]
                        #print b
        #print minDeg,b
        sum_a_x=sum_a_x+(minDeg*b)
        sum_a=sum_a+minDeg
    #print sum_a_x
    #print sum_a
    #print sum_a_x/sum_a
    return abs((test_data[4]-(sum_a_x/sum_a))/test_data[4])*100
    #return abs((test_data[4]-(sum_a_x/sum_a)))
    #return  sum_a
y=0
z=0
df1 = pd.read_csv('ele-3.csv')
for row in df1.iterrows():
    index, data = row
    #print data.tolist()
    x=MSE(data.tolist())
    y=y+x
    z=z+1
    #temp.append(data.tolist())

print y/z
print len(refined_rules_list),"Done"



#check both region
#x1_reg= range_finder(x1_points,x1)
#abc will be found here
#print x1_points[4],type(x1_points[4]) #show rajesh

#writing 50 refined  rule-base in file


# mapping ={'S2':0,'S1':1,'CE':2,'B1':3,'B2':4}
# f = open('refined_rule.dat', 'w')
# for i in range(len(refined_rules_list)):
#     ls = refined_rules_list[i][0:5]
#     for _ in ls:
#         f.write(str(mapping[_]))
#
#     f.write("\n")
#
# f =open('store_b','w')
# for i in range(5):
#     f.write(str(x1_a_b_c[i][2] )+" "+str(x2_a_b_c[i][2])+" "+str(x3_a_b_c[i][2])+" "+str(x4_a_b_c[i][2])+" "+str(y_a_b_c[i][2])+"\n")
# f=open('x1_a_b_c','w')
# for i in range(len(x1_a_b_c)):
#     f.write(str(mapping[x1_a_b_c[i][0]])+" "+str(x1_a_b_c[i][1])+" "+str(x1_a_b_c[i][2])+" "+str(x1_a_b_c[i][3])+"\n")
#
# f=open('x2_a_b_c','w')
# for i in range(len(x2_a_b_c)):
#     f.write(str(mapping[x2_a_b_c[i][0]])+" "+str(x2_a_b_c[i][1])+" "+str(x2_a_b_c[i][2])+" "+str(x2_a_b_c[i][3])+"\n")
#
# f=open('x3_a_b_c','w')
# for i in range(len(x3_a_b_c)):
#     f.write(str(mapping[x3_a_b_c[i][0]])+" "+str(x3_a_b_c[i][1])+" "+str(x3_a_b_c[i][2])+" "+str(x3_a_b_c[i][3])+"\n")
#
# f=open('x4_a_b_c','w')
# for i in range(len(x4_a_b_c)):
#     f.write(str(mapping[x4_a_b_c[i][0]])+" "+str(x4_a_b_c[i][1])+" "+str(x4_a_b_c[i][2])+" "+str(x4_a_b_c[i][3])+"\n")
#
# f=open('y_a_b_c','w')
# for i in range(len(y_a_b_c)):
#     f.write(str(mapping[y_a_b_c[i][0]])+" "+str(y_a_b_c[i][1])+" "+str(y_a_b_c[i][2])+" "+str(y_a_b_c[i][3])+"\n")



#GA starts here
#print x1_a_b_c
#print x2_a_b_c



# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes
x1_m_a_b_c=x1_a_b_c
x2_m_a_b_c=x2_a_b_c
x3_m_a_b_c=x3_a_b_c
x4_m_a_b_c=x4_a_b_c

def Update_abc(lst):

    x1_mod=lst[0:5]
    x2_mod=lst[5:10]
    x3_mod=lst[10:15]
    x4_mod=lst[15:20]
    b=0.03
    #print x1_mod
    for i in range(5):
        #print i
        if(x1_mod[i] == 0):
            x1_m_a_b_c[i][1] = x1_a_b_c[i][1] - b
            x1_m_a_b_c[i][2] = x1_a_b_c[i][2] - b
            x1_m_a_b_c[i][3] = x1_a_b_c[i][3] - b
        if(x1_mod[i] == 1):
            x1_m_a_b_c[i][1]=  x1_a_b_c[i][1] + b
            x1_m_a_b_c[i][2]=  x1_a_b_c[i][2] + b
            x1_m_a_b_c[i][3]=  x1_a_b_c[i][3] + b

    for i in range(5):
        if(x2_mod[i] == 0):
            x2_m_a_b_c[i][1]=x2_a_b_c[i][1] - b
            x2_m_a_b_c[i][2]=x2_a_b_c[i][2] - b
            x2_m_a_b_c[i][3]=x2_a_b_c[i][3] - b
        if(x2_mod[i] == 1):
            x2_m_a_b_c[i][1]=x2_a_b_c[i][1] + b
            x2_m_a_b_c[i][2]=x2_a_b_c[i][2] + b
            x2_m_a_b_c[i][3]=x2_a_b_c[i][3] + b

    for i in range(5):
        if(x3_mod[i] == 0):
            x3_m_a_b_c[i][1] = x3_a_b_c[i][1]  -b
            x3_m_a_b_c[i][2] = x3_a_b_c[i][2] - b
            x3_m_a_b_c[i][3] = x3_a_b_c[i][3] - b
        if(x3_mod[i] == 1):
            x3_m_a_b_c[i][1] = x3_a_b_c[i][1] + b
            x3_m_a_b_c[i][2] = x3_a_b_c[i][2] + b
            x3_m_a_b_c[i][3] = x3_a_b_c[i][3] + b

    for i in range(5):
        if(x4_mod[i] == 0):
            x4_m_a_b_c[i][1] = x4_a_b_c[i][1]-b
            x4_m_a_b_c[i][2] = x4_a_b_c[i][2]-b
            x4_m_a_b_c[i][3] = x4_a_b_c[i][3]-b
        if(x4_mod[i] == 1):
            x4_m_a_b_c[i][1] = x4_a_b_c[i][1]+b
            x4_m_a_b_c[i][2] = x4_a_b_c[i][2]+b
            x4_m_a_b_c[i][3] = x4_a_b_c[i][3]+b


def MOD_MSE(test_data):
    sum_a=0.0001
    sum_a_x=0
    b=0
    for i in range(50):

        minDeg=1.0
        for j in range(5):
            if(j==0):

                for k in range(4):
                    if(x1_a_b_c[k][0]==refined_rules_list[i][0]):
                        x=mfDegree(test_data[0],x1_m_a_b_c[k][1],x1_m_a_b_c[k][2],x1_m_a_b_c[k][3])
                        minDeg=min(x,minDeg)

            if(j==1):
                for k in range(4):
                    if(x2_a_b_c[k][0]==refined_rules_list[i][1]):
                        x= mfDegree(test_data[1],x2_m_a_b_c[k][1],x2_m_a_b_c[k][2],x2_m_a_b_c[k][3])
                        minDeg=min(x,minDeg)

            if(j==2):
                for k in range(4):
                    if(x3_a_b_c[k][0]==refined_rules_list[i][2]):
                        x= mfDegree(test_data[2],x3_m_a_b_c[k][1],x3_m_a_b_c[k][2],x3_m_a_b_c[k][3])
                        minDeg=min(x,minDeg)

            if(j==3):
                for k in range(4):
                    if(x4_a_b_c[k][0]==refined_rules_list[i][3]):
                        x= mfDegree(test_data[3],x4_m_a_b_c[k][1],x4_m_a_b_c[k][2],x4_m_a_b_c[k][3])

                        minDeg=min(x,minDeg)


            if(j==4):
                for k in range(4):
                    #print y_a_b_c[k][0],refined_rules_list[i][4]
                    if(y_a_b_c[k][0]==refined_rules_list[i][4]):
                        b= y_a_b_c[k][2]

        sum_a_x=sum_a_x+(minDeg*b)
        sum_a=sum_a+minDeg
    return abs(100-(abs(test_data[4]-(sum_a_x/sum_a))/test_data[4])*100)

    #return abs(test_data[4]-(sum_a_x/sum_a))


    #print "test data: " ,test_data[4]
    #print "sum_a_x: ",sum_a_x
    #print "sum_a : ",sum_a
    #return  sum_a




def eval_func(chromosome):
    score = 0.0
    count = 0.0
    y=0
    lst = list(chromosome)
    print ' ', lst
    Update_abc(lst)
    #iterate over the chromosome
    for row in df1.iterrows():
        index, data = row
        #print data.tolist()
        x = MOD_MSE(data.tolist())
        y=y+x
        count=count+1
    #temp.append(data.tolist())
    print y/count
    return float(y/count)

# # Genome instance
# genome = G1DBinaryString.G1DBinaryString(20)
#
# # The evaluator function (objective function)
# genome.evaluator.set(eval_func)
# genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)
#
# # Genetic Algorithm Instance
# ga = GSimpleGA.GSimpleGA(genome)
# ga.selector.set(Selectors.GTournamentSelector)
# ga.setPopulationSize(5)
# ga.setGenerations(20)
# #best = ga.bestIndividual()
# #min_score = best.score
#
# # Do the evolution, with stats dump
# # frequency of 10 generations
# ga.evolve(freq_stats=1)
#
# # Best individual
# best = ga.bestIndividual()
# #print type(ga.bestIndividual())
print 'Done'
