# test householder
# example https://www.youtube.com/watch?v=OqgYYqy0M4w

##A = [
##        1 ,        0 ,        0 ,        0 ,        0 ,        0 ;
##        3 ,        2 ,        0 ,        0 ,        0 ,        0 ;
##        5 ,        4 ,        3 ,        0 ,        0 ,        0 ;
##        7 ,        6 ,        5 ,        4 ,        0 ,        0 ;
##        9 ,        8 ,        7 ,        6 ,        5 ,        0 ;
##       11 ,       10 ,        9 ,        8 ,        7 ,        6 ;
##       13 ,       12 ,       11 ,       10 ,        9 ,        8 ;
##       15 ,       14 ,       13 ,       12 ,       11 ,       10 ;
##       17 ,       16 ,       15 ,       14 ,       13 ,       12 ;
##       19 ,       18 ,       17 ,       16 ,       15 ,       14 ;
##       21 ,       20 ,       19 ,       18 ,       17 ,       16 ;
##       23 ,       22 ,       21 ,       20 ,       19 ,       18 ;
##       25 ,       24 ,       23 ,       22 ,       21 ,       20 ;
##       27 ,       26 ,       25 ,       24 ,       23 ,       22 ;
##       29 ,       28 ,       27 ,       26 ,       25 ,       24 ;
##       31 ,       30 ,       29 ,       28 ,       27 ,       26 ;
##       33 ,       32 ,       31 ,       30 ,       29 ,       28 ;
##       35 ,       34 ,       33 ,       32 ,       31 ,       30 ;
##       37 ,       36 ,       35 ,       34 ,       33 ,       32 ;
##       39 ,       38 ,       37 ,       36 ,       35 ,       34 ;
##]

clear
clc

A = [
-1, -1, 1;
1, 3, 3;
-1, -1, 5]



[Q,R] = qr(A)
