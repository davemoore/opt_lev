////////////////////////////////////////////////// 
// gmsh geometry specification for cube
// homer reid
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// geometric parameters 
//////////////////////////////////////////////////
aspect = 2;
L = aspect*10;   // side length
D = 10; //depth
H = 10;

//////////////////////////////////////////////////
// this factor may be increased or decreased to   
// make the meshing more coarse or more fine in a
// uniform way over the entire object 
//////////////////////////////////////////////////
Mesh.CharacteristicLengthFactor=1.3;

//////////////////////////////////////////////////
// these factors may be configured separately
// to make the meshing more coarse or more fine in
// particular regions of the object 
//////////////////////////////////////////////////
grid = DefineNumber[ 0.2, Name "Parameters/grid" ];
lCoarse =  grid*30;
lFine   =  grid;

//////////////////////////////////////////////////
// geometric description of cube /////////////////
//////////////////////////////////////////////////
Point(1) = { L/2, -H/2, -D, lCoarse};
Point(2) = {-L/2, -H/2, -D, lCoarse};
Point(3) = {-L/2, -H/2,  0, lFine};
Point(4) = { L/2, -H/2,  0, lFine};

Point(5) = { 0, -H/2, 0, grid};
Point(6) = { 0, H/2, 0, grid};

Point(7) = { L/2, -H/2, -D, lCoarse};
Point(8) = {-L/2, -H/2, -D, lCoarse};
Point(9) = {-L/2, -H/2,  0, lFine};
Point(10) = { L/2, -H/2,  0, lFine};