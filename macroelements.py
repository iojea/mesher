import numpy as np

mu = .65
hybrid_color    = "green"
isotropic_color = "red"
prismatic_color = "blue"

colors = [ "green", "red", "blue"]

x_max = 3
x_int_max = 2
x_min   = -1
x_int_min   = 0
y_max   = 1
y_min   = -3
y_int_min   = -2
y_int_max   = 0
z_max   = 0

with open('partition.txt','r') as infile:
    inlist = infile.readlines()

pre_list = [line.strip(' \n').split(',') for line in inlist]
pre_list = [[int(st[0])] + [int(st[k]) for k in xrange(1,len(st)-1)]+[float(st[-1])] for st in pre_list]

macro_elements_aux = { key : 
                        { 0 : np.array(pre_list[key][1:-1]).reshape((len(pre_list[key])-2)/3,3), 
                          1 : pre_list[key][-1], 
                          2 : colors[pre_list[key][0]], 
                          3 : pre_list[key][0] } 
                       for key in range(len(pre_list)) }

macro_elements = { 
0 : { 0 : np.array([[x_int_min,y_int_max,-1],[-1,0,-1],[0,1,-1],[0,0,0]]), 1 : mu, 2 : hybrid_color,    3 : 0},
1 : { 0 : np.array([[-1,0,0],[-1,1,0],[-1,0,-1],[0,0,0]]),                 1 : mu, 2 : hybrid_color,    3 : 0}, 
2 : { 0 : np.array([[0,1,0],[0,1,-1],[-1,1,0],[0,0,0]]),                   1 : mu, 2 : hybrid_color,    3 : 0},
3 : { 0 : np.array([[-1,1,-1],[-1,0,-1],[-1,1,0],[0,1,-1]]),               1 : 1,  2 : hybrid_color,    3 : 0},
4 : { 0 : np.array([[0,0,0],[-1,0,-1],[-1,1,0],[0,1,-1]]),                 1 : mu, 2 : isotropic_color, 3 : 1},

5 : { 0 : np.array([-1,1,1])*np.array([[x_int_min,y_int_max,-1],[-1,0,-1],[0,1,-1],[0,0,0]])  + np.array([2,0,0]), 1 : mu,2 : hybrid_color,    3 : 0},
6 : { 0 : np.array([-1,1,1])*np.array([[-1,0,0],[-1,1,0],[-1,0,-1],[0,0,0]])                  + np.array([2,0,0]), 1 : mu,2 : hybrid_color,    3 : 0},
7 : { 0 : np.array([-1,1,1])*np.array([[0,1,0],[0,1,-1],[-1,1,0],[0,0,0]])                    + np.array([2,0,0]), 1 : mu,2 : hybrid_color,    3 : 0},
8 : { 0 : np.array([-1,1,1])*np.array([[-1,1,-1],[-1,0,-1],[-1,1,0],[0,1,-1]])                + np.array([2,0,0]), 1 : 1 ,2 : hybrid_color,    3 : 0},
9 : { 0 : np.array([-1,1,1])*np.array([[0,0,0],[-1,0,-1],[-1,1,0],[0,1,-1]])                  + np.array([2,0,0]), 1 : mu,2 : isotropic_color, 3 : 1},

10 : {0: np.array([[0,-2,-1],[-1,-2,-1],[0,-3,-1],[0,-2,z_max]])        ,  1 : mu,2 : hybrid_color,    3 : 0},
11 : {0: np.array([[0,-3,0],[0,-3,-1],[-1,-3,0],[0,-2,z_max]])          ,  1 : mu,2 : hybrid_color,    3 : 0},
12 : {0: np.array([[-1,-2,z_max],[-1,-3,z_max],[-1,-2,-1],[0,-2,z_max]]),  1 : mu,2 : hybrid_color,    3 : 0},
13 : {0: np.array([[-1,-3,-1],[-1,-3,z_max],[0,-3,-1],[-1,-2,-1]])      ,  1 : 1 ,2 : hybrid_color,    3 : 0},
14 : {0: np.array([[ 0, -2,  0],[-1, -2, -1],[ 0, -3, -1],[-1, -3,  0]]),  1 : mu,2 : isotropic_color, 3 : 1},

15 : {0 : np.array([[x_int_min,-1,z_max],[x_int_min,-1,-1],[x_min,-1,z_max],[x_int_min, y_int_max,z_max]])   ,  1 : mu,2 : hybrid_color,    3 : 0},
16 : {0 : np.array([[x_int_min,y_int_max,-1],[x_min,y_int_max,-1],[x_int_min,-1,-1],[x_int_min,y_int_max,0]]),  1 : mu,2 : hybrid_color,    3 : 0},
17 : {0 : np.array([[-1,0,0],[-1,-1,0],[-1,0,-1],[0,0,0]])                                                   ,  1 : mu,2 : hybrid_color,    3 : 0},
18 : {0 : np.array([[-1,-1,-1],[-1,-1,0],[0,-1,-1],[-1,0,-1]])                                               ,  1 : 1 ,2 : hybrid_color,    3 : 0},
19 : {0 : np.array([[x_int_min,y_int_max,z_max],[-1,-1,z_max],[x_min,y_int_max,-1],[x_int_min,-1,-1]])       ,  1 : mu,2 : isotropic_color, 3 : 1},

20 : {0 : np.array([[x_int_min,y_int_min,-1],[x_int_min,-1,-1],[x_min,y_int_min,-1],[x_int_min,y_int_min,0]]),    1 : mu,2 : hybrid_color,    3 : 0},
21 : {0 : np.array([[x_int_min, -1, z_max],[x_min, -1, z_max],[x_int_min, -1, -1],[x_int_min, y_int_min, z_max]]),1 : mu,2 : hybrid_color,    3 : 0},
22 : {0 : np.array([[-1,-2,0],[-1,-2,-1],[-1,-1,0],[0,-2,0]]) ,                                                   1 : mu,2 : hybrid_color,    3 : 0},
23 : {0 : np.array([[-1,-1,-1],[0,-1,-1],[-1,-1,0],[-1,-2,-1]]),                                                  1 : 1 ,2 : hybrid_color,    3 : 0},
24 : {0 : np.array([[x_int_min,y_int_min,z_max],[-1,-1,z_max],[x_int_min,-1,-1],[x_min,y_int_min,-1]]),           1 : mu,2 : isotropic_color, 3 : 1},

25 : {0 : np.array([[ 2,-2,-1],[ 3,-2,-1],[ 2,-3,-1],[ 2,-2, 0]]),1 : mu,2 : hybrid_color,    3 : 0},
26 : {0 : np.array([[ 2,-3, 0],[ 2,-3,-1],[ 3,-3, 0],[ 2,-2, 0]]),1 : mu,2 : hybrid_color,    3 : 0},
27 : {0 : np.array([[ 3,-2, 0],[ 3,-3, 0],[ 3,-2,-1],[ 2,-2, 0]]),1 : mu,2 : hybrid_color,    3 : 0},
28 : {0 : np.array([[ 3,-3,-1],[ 3,-3, 0],[ 2,-3,-1],[ 3,-2,-1]]),1 : 1 ,2 : hybrid_color,    3 : 0},
29 : {0 : np.array([[ 2,-2, 0],[ 3,-2,-1],[ 2,-3,-1],[ 3,-3, 0]]),1 : mu,2 : isotropic_color, 3 : 1},

30 : {0: np.array([[2,-1, 0],[2,-1,-1],[3,-1, 0],[2,0,0]]),1 : mu,2 : hybrid_color,    3 : 0},
31 : {0: np.array([[2,0,-1],[3,0,-1],[2,-1,-1],[2,0,0]]),1 : mu,2 : hybrid_color,    3 : 0},
32: {0: np.array([[3,0,0],[3,-1,0],[3,0,-1],[2,0,0]]),1 : mu,2 : hybrid_color,    3 : 0},
33 : {0: np.array([[3,-1,-1],[3,-1,0],[2,-1,-1],[3,0,-1]]),1 : 1 ,2 : hybrid_color,    3 : 0},
34 : {0: np.array([[2,0,0],[3,-1,0],[3,0,-1],[2,-1,-1]]),1 : mu,2 : isotropic_color, 3 : 1},

35 : {0 : np.array([[2,-2,-1],[2,-1,-1],[3,-2,-1],[2,-2, 0]]),1 : mu,2 : hybrid_color,    3 : 0},
36 : {0 : np.array([[2,-1, 0],[3,-1, 0],[2,-1,-1],[2,-2, 0]]),1 : mu,2 : hybrid_color,    3 : 0},
37 : {0 : np.array([[3,-2, 0],[3,-2,-1],[3,-1, 0],[2,-2, 0]]),1 : mu,2 : hybrid_color,    3 : 0},
38 : {0 : np.array([[3,-1,-1],[2,-1,-1],[3,-1, 0],[3,-2,-1]]),1 : 1 ,2 : hybrid_color,    3 : 0},
39 : {0 : np.array([[2,-2, 0],[3,-1, 0],[2,-1,-1],[3,-2,-1]]),1 : mu,2 : isotropic_color, 3 : 1},

40 : {0 : np.array([[ 0,  0, -1],[ 0,  1, -1],[-1,  0, -1],[ 0,  0, -5],[ 0,  1, -5],[-1,  0, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
41 : {0 : np.array([[-1,  1, -1],[ 0,  1, -1],[-1,  0, -1],[-1,  1, -5],[ 0,  1, -5],[-1,  0, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
42 : {0 : np.array([[-1, -3, -1],[ 0, -3, -1],[-1, -2, -1],[-1, -3, -5],[ 0, -3, -5],[-1, -2, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
43 : {0 : np.array([[ 0, -2, -1],[-1, -2, -1],[ 0, -3, -1],[ 0, -2, -5],[-1, -2, -5],[ 0, -3, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
44 : {0 : np.array([[ 0, -2, -1],[ 0, -1, -1],[-1, -2, -1],[ 0, -2, -5],[ 0, -1, -5],[ 0, -2, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
45 : {0 : np.array([[ 0, -2, -1],[ 0, -3, -1],[ 1, -2, -1],[ 0, -2, -5],[ 0, -3, -5],[ 1, -2, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
46 : {0 : np.array([[-1, -1, -1],[-1, -2, -1],[ 0, -1, -1],[-1, -1, -5],[-1, -2, -5],[ 0, -1, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
47 : {0 : np.array([[ 1, -3, -1],[ 1, -2, -1],[ 0, -3, -1],[ 1, -3, -5],[ 1, -2, -5],[ 0, -3, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
48 : {0 : np.array([[ 0,  0, -1],[-1,  0, -1],[ 0, -1, -1],[ 0,  0, -5],[-1,  0, -5],[ 0, -1, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
49 : {0 : np.array([[-1, -1, -1],[ 0, -1, -1],[-1,  0, -1],[-1, -1, -5],[ 0, -1, -5],[-1,  0, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
50 : {0 : np.array([[ 1, -3, -1],[ 2, -3, -1],[ 1, -2, -1],[ 1, -3, -5],[ 2, -3, -5],[ 1, -2, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
51 : {0 : np.array([[ 2, -2, -1],[ 1, -2, -1],[ 2, -3, -1],[ 2, -2, -5],[ 1, -2, -5],[ 2, -3, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
52 : {0 : np.array([[ 2, -2, -1],[ 2, -3, -1],[ 3, -2, -1],[ 2, -2, -5],[ 2, -3, -5],[ 3, -2, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
53 : {0 : np.array([[ 3, -3, -1],[ 3, -2, -1],[ 2, -3, -1],[ 3, -3, -5],[ 3, -2, -5],[ 2, -3, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
54 : {0 : np.array([[ 2, -2, -1],[ 3, -2, -1],[ 2, -1, -1],[ 2, -2, -5],[ 3, -2, -5],[ 2, -1, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
55 : {0 : np.array([[ 3, -1, -1],[ 2, -1, -1],[ 3, -2, -1],[ 3, -1, -5],[ 2, -1, -5],[ 3, -2, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
56 : {0 : np.array([[ 3, -1, -1],[ 3,  0, -1],[ 2, -1, -1],[ 3, -1, -5],[ 3,  0, -5],[ 2, -1, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
57 : {0 : np.array([[ 2,  0, -1],[ 2, -1, -1],[ 3,  0, -1],[ 2,  0, -5],[ 2, -1, -5],[ 3,  0, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
58 : {0 : np.array([[ 2,  0, -1],[ 3,  0, -1],[ 2,  1, -1],[ 2,  0, -5],[ 3,  0, -5],[ 2,  1, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
59 : {0 : np.array([[ 3,  1, -1],[ 2,  1, -1],[ 3,  0, -1],[ 3,  1, -5],[ 2,  1, -5],[ 3,  0, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
60 : {0 : np.array([[ 2,  0, -1],[ 2,  1, -1],[ 1,  0, -1],[ 2,  0, -5],[ 2,  1, -5],[ 1,  0, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2},
61 : {0 : np.array([[ 1,  1, -1],[ 1,  0, -1],[ 2,  1, -1],[ 1,  1, -5],[ 1,  0, -5],[ 2,  1, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
62 : {0 : np.array([[ 1,  1, -1],[ 0,  1, -1],[ 1,  0, -1],[ 1,  1, -5],[ 0,  1, -5],[ 1,  0, -5]]), 1 : 1, 2 : prismatic_color, 3 : 2},
63 : {0 : np.array([[ 0,  0, -1],[ 1,  0, -1],[ 0,  1, -1],[ 0,  0, -5],[ 1,  0, -5],[ 0,  1, -5]]), 1 : mu, 2 : prismatic_color, 3 : 2}
}
