??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_44/kernel
}
$conv2d_44/kernel/Read/ReadVariableOpReadVariableOpconv2d_44/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_44/bias
m
"conv2d_44/bias/Read/ReadVariableOpReadVariableOpconv2d_44/bias*
_output_shapes
:@*
dtype0
?
conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_45/kernel
~
$conv2d_45/kernel/Read/ReadVariableOpReadVariableOpconv2d_45/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_45/bias
n
"conv2d_45/bias/Read/ReadVariableOpReadVariableOpconv2d_45/bias*
_output_shapes	
:?*
dtype0
?
conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_46/kernel

$conv2d_46/kernel/Read/ReadVariableOpReadVariableOpconv2d_46/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_46/bias
n
"conv2d_46/bias/Read/ReadVariableOpReadVariableOpconv2d_46/bias*
_output_shapes	
:?*
dtype0
?
conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_47/kernel

$conv2d_47/kernel/Read/ReadVariableOpReadVariableOpconv2d_47/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_47/bias
n
"conv2d_47/bias/Read/ReadVariableOpReadVariableOpconv2d_47/bias*
_output_shapes	
:?*
dtype0
z
dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H* 
shared_namedense_63/kernel
s
#dense_63/kernel/Read/ReadVariableOpReadVariableOpdense_63/kernel*
_output_shapes

:H*
dtype0
r
dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_63/bias
k
!dense_63/bias/Read/ReadVariableOpReadVariableOpdense_63/bias*
_output_shapes
:H*
dtype0
z
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D* 
shared_namedense_61/kernel
s
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes

:D*
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
:D*
dtype0
|
dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?(?* 
shared_namedense_64/kernel
u
#dense_64/kernel/Read/ReadVariableOpReadVariableOpdense_64/kernel* 
_output_shapes
:
?(?*
dtype0
s
dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_64/bias
l
!dense_64/bias/Read/ReadVariableOpReadVariableOpdense_64/bias*
_output_shapes	
:?*
dtype0
|
dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?$?* 
shared_namedense_62/kernel
u
#dense_62/kernel/Read/ReadVariableOpReadVariableOpdense_62/kernel* 
_output_shapes
:
?$?*
dtype0
s
dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_62/bias
l
!dense_62/bias/Read/ReadVariableOpReadVariableOpdense_62/bias*
_output_shapes	
:?*
dtype0
|
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_60/kernel
u
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel* 
_output_shapes
:
??*
dtype0
s
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_60/bias
l
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes	
:?*
dtype0
|
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_65/kernel
u
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel* 
_output_shapes
:
??*
dtype0
s
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_65/bias
l
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
_output_shapes	
:?*
dtype0
|
dense_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_66/kernel
u
#dense_66/kernel/Read/ReadVariableOpReadVariableOpdense_66/kernel* 
_output_shapes
:
??*
dtype0
s
dense_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_66/bias
l
!dense_66/bias/Read/ReadVariableOpReadVariableOpdense_66/bias*
_output_shapes	
:?*
dtype0
{
dense_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_67/kernel
t
#dense_67/kernel/Read/ReadVariableOpReadVariableOpdense_67/kernel*
_output_shapes
:	?*
dtype0
r
dense_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_67/bias
k
!dense_67/bias/Read/ReadVariableOpReadVariableOpdense_67/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_44/kernel/m
?
+Adam/conv2d_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_44/bias/m
{
)Adam/conv2d_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_45/kernel/m
?
+Adam/conv2d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_45/bias/m
|
)Adam/conv2d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_46/kernel/m
?
+Adam/conv2d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_46/bias/m
|
)Adam/conv2d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_47/kernel/m
?
+Adam/conv2d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_47/bias/m
|
)Adam/conv2d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*'
shared_nameAdam/dense_63/kernel/m
?
*Adam/dense_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/m*
_output_shapes

:H*
dtype0
?
Adam/dense_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_63/bias/m
y
(Adam/dense_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/m*
_output_shapes
:H*
dtype0
?
Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D*'
shared_nameAdam/dense_61/kernel/m
?
*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m*
_output_shapes

:D*
dtype0
?
Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
:D*
dtype0
?
Adam/dense_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?(?*'
shared_nameAdam/dense_64/kernel/m
?
*Adam/dense_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/m* 
_output_shapes
:
?(?*
dtype0
?
Adam/dense_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_64/bias/m
z
(Adam/dense_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?$?*'
shared_nameAdam/dense_62/kernel/m
?
*Adam/dense_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/m* 
_output_shapes
:
?$?*
dtype0
?
Adam/dense_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_62/bias/m
z
(Adam/dense_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_60/kernel/m
?
*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_60/bias/m
z
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_65/kernel/m
?
*Adam/dense_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_65/bias/m
z
(Adam/dense_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_66/kernel/m
?
*Adam/dense_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_66/bias/m
z
(Adam/dense_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_67/kernel/m
?
*Adam/dense_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_67/bias/m
y
(Adam/dense_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_44/kernel/v
?
+Adam/conv2d_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_44/bias/v
{
)Adam/conv2d_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_45/kernel/v
?
+Adam/conv2d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_45/bias/v
|
)Adam/conv2d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_46/kernel/v
?
+Adam/conv2d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_46/bias/v
|
)Adam/conv2d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_47/kernel/v
?
+Adam/conv2d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_47/bias/v
|
)Adam/conv2d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*'
shared_nameAdam/dense_63/kernel/v
?
*Adam/dense_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/v*
_output_shapes

:H*
dtype0
?
Adam/dense_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_63/bias/v
y
(Adam/dense_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/v*
_output_shapes
:H*
dtype0
?
Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D*'
shared_nameAdam/dense_61/kernel/v
?
*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v*
_output_shapes

:D*
dtype0
?
Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
:D*
dtype0
?
Adam/dense_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?(?*'
shared_nameAdam/dense_64/kernel/v
?
*Adam/dense_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/v* 
_output_shapes
:
?(?*
dtype0
?
Adam/dense_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_64/bias/v
z
(Adam/dense_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?$?*'
shared_nameAdam/dense_62/kernel/v
?
*Adam/dense_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/v* 
_output_shapes
:
?$?*
dtype0
?
Adam/dense_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_62/bias/v
z
(Adam/dense_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_60/kernel/v
?
*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_60/bias/v
z
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_65/kernel/v
?
*Adam/dense_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_65/bias/v
z
(Adam/dense_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_66/kernel/v
?
*Adam/dense_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_66/bias/v
z
(Adam/dense_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_67/kernel/v
?
*Adam/dense_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_67/bias/v
y
(Adam/dense_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Қ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer_with_weights-8
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer_with_weights-9
layer-27
layer_with_weights-10
layer-28
layer_with_weights-11
layer-29
	optimizer
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$
signatures
 
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
R
+	variables
,trainable_variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
R
9	variables
:trainable_variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
R
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
R
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
 
 
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
h

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
R
a	variables
btrainable_variables
cregularization_losses
d	keras_api
R
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
R
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
R
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
h

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
h

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
k

}kernel
~bias
	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate%m?&m?/m?0m?=m?>m?Km?Lm?Um?Vm?[m?\m?qm?rm?wm?xm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?%v?&v?/v?0v?=v?>v?Kv?Lv?Uv?Vv?[v?\v?qv?rv?wv?xv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?
?
%0
&1
/2
03
=4
>5
K6
L7
U8
V9
[10
\11
q12
r13
w14
x15
}16
~17
?18
?19
?20
?21
?22
?23
?
%0
&1
/2
03
=4
>5
K6
L7
U8
V9
[10
\11
q12
r13
w14
x15
}16
~17
?18
?19
?20
?21
?22
?23
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
 
\Z
VARIABLE_VALUEconv2d_44/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_44/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
\Z
VARIABLE_VALUEconv2d_45/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_45/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
\Z
VARIABLE_VALUEconv2d_46/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_46/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
\Z
VARIABLE_VALUEconv2d_47/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_47/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
[Y
VARIABLE_VALUEdense_63/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_63/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[Y
VARIABLE_VALUEdense_61/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_61/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
[Y
VARIABLE_VALUEdense_64/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_64/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
[Y
VARIABLE_VALUEdense_62/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_62/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

w0
x1

w0
x1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
[Y
VARIABLE_VALUEdense_60/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_60/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1

}0
~1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
[Y
VARIABLE_VALUEdense_65/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_65/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
\Z
VARIABLE_VALUEdense_66/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_66/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
\Z
VARIABLE_VALUEdense_67/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_67/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_44/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_44/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_45/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_63/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_63/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_61/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_64/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_64/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_62/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_62/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_60/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_60/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_65/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_66/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_66/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_67/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_67/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_44/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_44/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_45/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_63/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_63/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_61/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_61/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_64/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_64/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_62/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_62/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_60/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_60/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_65/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_66/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_66/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_67/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_67/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_ConvInputPlaceholder*/
_output_shapes
:?????????00*
dtype0*$
shape:?????????00
?
serving_default_HogInputPlaceholder*+
_output_shapes
:?????????H*
dtype0* 
shape:?????????H
?
serving_default_LandmarksInputPlaceholder*+
_output_shapes
:?????????D*
dtype0* 
shape:?????????D
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_ConvInputserving_default_HogInputserving_default_LandmarksInputconv2d_44/kernelconv2d_44/biasconv2d_45/kernelconv2d_45/biasconv2d_46/kernelconv2d_46/biasconv2d_47/kernelconv2d_47/biasdense_61/kerneldense_61/biasdense_63/kerneldense_63/biasdense_60/kerneldense_60/biasdense_62/kerneldense_62/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_89002
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_44/kernel/Read/ReadVariableOp"conv2d_44/bias/Read/ReadVariableOp$conv2d_45/kernel/Read/ReadVariableOp"conv2d_45/bias/Read/ReadVariableOp$conv2d_46/kernel/Read/ReadVariableOp"conv2d_46/bias/Read/ReadVariableOp$conv2d_47/kernel/Read/ReadVariableOp"conv2d_47/bias/Read/ReadVariableOp#dense_63/kernel/Read/ReadVariableOp!dense_63/bias/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOp#dense_64/kernel/Read/ReadVariableOp!dense_64/bias/Read/ReadVariableOp#dense_62/kernel/Read/ReadVariableOp!dense_62/bias/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOp#dense_65/kernel/Read/ReadVariableOp!dense_65/bias/Read/ReadVariableOp#dense_66/kernel/Read/ReadVariableOp!dense_66/bias/Read/ReadVariableOp#dense_67/kernel/Read/ReadVariableOp!dense_67/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_44/kernel/m/Read/ReadVariableOp)Adam/conv2d_44/bias/m/Read/ReadVariableOp+Adam/conv2d_45/kernel/m/Read/ReadVariableOp)Adam/conv2d_45/bias/m/Read/ReadVariableOp+Adam/conv2d_46/kernel/m/Read/ReadVariableOp)Adam/conv2d_46/bias/m/Read/ReadVariableOp+Adam/conv2d_47/kernel/m/Read/ReadVariableOp)Adam/conv2d_47/bias/m/Read/ReadVariableOp*Adam/dense_63/kernel/m/Read/ReadVariableOp(Adam/dense_63/bias/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp*Adam/dense_64/kernel/m/Read/ReadVariableOp(Adam/dense_64/bias/m/Read/ReadVariableOp*Adam/dense_62/kernel/m/Read/ReadVariableOp(Adam/dense_62/bias/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp*Adam/dense_65/kernel/m/Read/ReadVariableOp(Adam/dense_65/bias/m/Read/ReadVariableOp*Adam/dense_66/kernel/m/Read/ReadVariableOp(Adam/dense_66/bias/m/Read/ReadVariableOp*Adam/dense_67/kernel/m/Read/ReadVariableOp(Adam/dense_67/bias/m/Read/ReadVariableOp+Adam/conv2d_44/kernel/v/Read/ReadVariableOp)Adam/conv2d_44/bias/v/Read/ReadVariableOp+Adam/conv2d_45/kernel/v/Read/ReadVariableOp)Adam/conv2d_45/bias/v/Read/ReadVariableOp+Adam/conv2d_46/kernel/v/Read/ReadVariableOp)Adam/conv2d_46/bias/v/Read/ReadVariableOp+Adam/conv2d_47/kernel/v/Read/ReadVariableOp)Adam/conv2d_47/bias/v/Read/ReadVariableOp*Adam/dense_63/kernel/v/Read/ReadVariableOp(Adam/dense_63/bias/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOp*Adam/dense_64/kernel/v/Read/ReadVariableOp(Adam/dense_64/bias/v/Read/ReadVariableOp*Adam/dense_62/kernel/v/Read/ReadVariableOp(Adam/dense_62/bias/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOp*Adam/dense_65/kernel/v/Read/ReadVariableOp(Adam/dense_65/bias/v/Read/ReadVariableOp*Adam/dense_66/kernel/v/Read/ReadVariableOp(Adam/dense_66/bias/v/Read/ReadVariableOp*Adam/dense_67/kernel/v/Read/ReadVariableOp(Adam/dense_67/bias/v/Read/ReadVariableOpConst*^
TinW
U2S	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_90303
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_44/kernelconv2d_44/biasconv2d_45/kernelconv2d_45/biasconv2d_46/kernelconv2d_46/biasconv2d_47/kernelconv2d_47/biasdense_63/kerneldense_63/biasdense_61/kerneldense_61/biasdense_64/kerneldense_64/biasdense_62/kerneldense_62/biasdense_60/kerneldense_60/biasdense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_44/kernel/mAdam/conv2d_44/bias/mAdam/conv2d_45/kernel/mAdam/conv2d_45/bias/mAdam/conv2d_46/kernel/mAdam/conv2d_46/bias/mAdam/conv2d_47/kernel/mAdam/conv2d_47/bias/mAdam/dense_63/kernel/mAdam/dense_63/bias/mAdam/dense_61/kernel/mAdam/dense_61/bias/mAdam/dense_64/kernel/mAdam/dense_64/bias/mAdam/dense_62/kernel/mAdam/dense_62/bias/mAdam/dense_60/kernel/mAdam/dense_60/bias/mAdam/dense_65/kernel/mAdam/dense_65/bias/mAdam/dense_66/kernel/mAdam/dense_66/bias/mAdam/dense_67/kernel/mAdam/dense_67/bias/mAdam/conv2d_44/kernel/vAdam/conv2d_44/bias/vAdam/conv2d_45/kernel/vAdam/conv2d_45/bias/vAdam/conv2d_46/kernel/vAdam/conv2d_46/bias/vAdam/conv2d_47/kernel/vAdam/conv2d_47/bias/vAdam/dense_63/kernel/vAdam/dense_63/bias/vAdam/dense_61/kernel/vAdam/dense_61/bias/vAdam/dense_64/kernel/vAdam/dense_64/bias/vAdam/dense_62/kernel/vAdam/dense_62/bias/vAdam/dense_60/kernel/vAdam/dense_60/bias/vAdam/dense_65/kernel/vAdam/dense_65/bias/vAdam/dense_66/kernel/vAdam/dense_66/bias/vAdam/dense_67/kernel/vAdam/dense_67/bias/v*]
TinV
T2R*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_90556??
?
?
'__inference_model_7_layer_call_fn_89112
inputs_0
inputs_1
inputs_2!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:D
	unknown_8:D
	unknown_9:H

unknown_10:H

unknown_11:
??

unknown_12:	?

unknown_13:
?$?

unknown_14:	?

unknown_15:
?(?

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	?

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_7_layer_call_and_return_conditional_losses_88671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????00
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????D
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
inputs/2
?	
d
E__inference_dropout_57_layer_call_and_return_conditional_losses_88285

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_62_layer_call_and_return_conditional_losses_89848

inputs2
matmul_readvariableop_resource:
?$?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
g
-__inference_concatenate_7_layer_call_fn_89956
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_88108a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?

d
E__inference_dropout_55_layer_call_and_return_conditional_losses_89628

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87803

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_65_layer_call_and_return_conditional_losses_88129

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_46_layer_call_fn_89591

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87891i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_62_layer_call_and_return_conditional_losses_88056

inputs2
matmul_readvariableop_resource:
?$?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_45_layer_call_fn_89524

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87861i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????

?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_89002
	convinput
hoginput
landmarksinput!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:D
	unknown_8:D
	unknown_9:H

unknown_10:H

unknown_11:
??

unknown_12:	?

unknown_13:
?$?

unknown_14:	?

unknown_15:
?(?

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	?

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	convinputlandmarksinputhoginputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_87758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????H:?????????D: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????00
#
_user_specified_name	ConvInput:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
HogInput:[W
+
_output_shapes
:?????????D
(
_user_specified_nameLandmarksInput
?
?
'__inference_model_7_layer_call_fn_88777
	convinput
landmarksinput
hoginput!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:D
	unknown_8:D
	unknown_9:H

unknown_10:H

unknown_11:
??

unknown_12:	?

unknown_13:
?$?

unknown_14:	?

unknown_15:
?(?

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	?

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	convinputlandmarksinputhoginputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_7_layer_call_and_return_conditional_losses_88671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????00
#
_user_specified_name	ConvInput:[W
+
_output_shapes
:?????????D
(
_user_specified_nameLandmarksInput:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
HogInput
?
c
E__inference_dropout_58_layer_call_and_return_conditional_losses_89910

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_7_layer_call_fn_88221
	convinput
landmarksinput
hoginput!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:D
	unknown_8:D
	unknown_9:H

unknown_10:H

unknown_11:
??

unknown_12:	?

unknown_13:
?$?

unknown_14:	?

unknown_15:
?(?

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	?

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	convinputlandmarksinputhoginputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_7_layer_call_and_return_conditional_losses_88170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????00
#
_user_specified_name	ConvInput:[W
+
_output_shapes
:?????????D
(
_user_specified_nameLandmarksInput:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
HogInput
?
c
*__inference_dropout_54_layer_call_fn_89544

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_54_layer_call_and_return_conditional_losses_88498x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????

?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????

?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?
F
*__inference_flatten_29_layer_call_fn_89791

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_29_layer_call_and_return_conditional_losses_88018a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????DD:S O
+
_output_shapes
:?????????DD
 
_user_specified_nameinputs
?
c
E__inference_dropout_59_layer_call_and_return_conditional_losses_88084

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_66_layer_call_and_return_conditional_losses_88146

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_89489

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_46_layer_call_fn_89586

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87791?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87921

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_66_layer_call_fn_90004

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_88146p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_63_layer_call_and_return_conditional_losses_89708

inputs3
!tensordot_readvariableop_resource:H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:H*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????H?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:HY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????HHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????HHT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????HHe
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????HHz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
C__inference_dense_61_layer_call_and_return_conditional_losses_89748

inputs3
!tensordot_readvariableop_resource:D-
biasadd_readvariableop_resource:D
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:D*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????D?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????D[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:DY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????DDr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:D*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????DDT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????DDe
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????DDz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????D
 
_user_specified_nameinputs
?
c
*__inference_dropout_58_layer_call_fn_89905

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_88308p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_64_layer_call_and_return_conditional_losses_89828

inputs2
matmul_readvariableop_resource:
?(?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?(?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87891

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_89601

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
B__inference_model_7_layer_call_and_return_conditional_losses_89454
inputs_0
inputs_1
inputs_2B
(conv2d_44_conv2d_readvariableop_resource:@7
)conv2d_44_biasadd_readvariableop_resource:@C
(conv2d_45_conv2d_readvariableop_resource:@?8
)conv2d_45_biasadd_readvariableop_resource:	?D
(conv2d_46_conv2d_readvariableop_resource:??8
)conv2d_46_biasadd_readvariableop_resource:	?D
(conv2d_47_conv2d_readvariableop_resource:??8
)conv2d_47_biasadd_readvariableop_resource:	?<
*dense_61_tensordot_readvariableop_resource:D6
(dense_61_biasadd_readvariableop_resource:D<
*dense_63_tensordot_readvariableop_resource:H6
(dense_63_biasadd_readvariableop_resource:H;
'dense_60_matmul_readvariableop_resource:
??7
(dense_60_biasadd_readvariableop_resource:	?;
'dense_62_matmul_readvariableop_resource:
?$?7
(dense_62_biasadd_readvariableop_resource:	?;
'dense_64_matmul_readvariableop_resource:
?(?7
(dense_64_biasadd_readvariableop_resource:	?;
'dense_65_matmul_readvariableop_resource:
??7
(dense_65_biasadd_readvariableop_resource:	?;
'dense_66_matmul_readvariableop_resource:
??7
(dense_66_biasadd_readvariableop_resource:	?:
'dense_67_matmul_readvariableop_resource:	?6
(dense_67_biasadd_readvariableop_resource:
identity?? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp? conv2d_46/BiasAdd/ReadVariableOp?conv2d_46/Conv2D/ReadVariableOp? conv2d_47/BiasAdd/ReadVariableOp?conv2d_47/Conv2D/ReadVariableOp?dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?!dense_61/Tensordot/ReadVariableOp?dense_62/BiasAdd/ReadVariableOp?dense_62/MatMul/ReadVariableOp?dense_63/BiasAdd/ReadVariableOp?!dense_63/Tensordot/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?dense_65/MatMul/ReadVariableOp?dense_66/BiasAdd/ReadVariableOp?dense_66/MatMul/ReadVariableOp?dense_67/BiasAdd/ReadVariableOp?dense_67/MatMul/ReadVariableOp?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_44/Conv2DConv2Dinputs_0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@*
paddingVALID*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@l
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:?????????,,@?
max_pooling2d_44/MaxPoolMaxPoolconv2d_44/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_45/Conv2DConv2D!max_pooling2d_44/MaxPool:output:0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_45/MaxPoolMaxPoolconv2d_45/Relu:activations:0*0
_output_shapes
:?????????

?*
ksize
*
paddingVALID*
strides
]
dropout_54/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_54/dropout/MulMul!max_pooling2d_45/MaxPool:output:0!dropout_54/dropout/Const:output:0*
T0*0
_output_shapes
:?????????

?i
dropout_54/dropout/ShapeShape!max_pooling2d_45/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_54/dropout/random_uniform/RandomUniformRandomUniform!dropout_54/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????

?*
dtype0f
!dropout_54/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_54/dropout/GreaterEqualGreaterEqual8dropout_54/dropout/random_uniform/RandomUniform:output:0*dropout_54/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????

??
dropout_54/dropout/CastCast#dropout_54/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????

??
dropout_54/dropout/Mul_1Muldropout_54/dropout/Mul:z:0dropout_54/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????

??
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_46/Conv2DConv2Ddropout_54/dropout/Mul_1:z:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_46/MaxPoolMaxPoolconv2d_46/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
]
dropout_55/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_55/dropout/MulMul!max_pooling2d_46/MaxPool:output:0!dropout_55/dropout/Const:output:0*
T0*0
_output_shapes
:??????????i
dropout_55/dropout/ShapeShape!max_pooling2d_46/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_55/dropout/random_uniform/RandomUniformRandomUniform!dropout_55/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0f
!dropout_55/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_55/dropout/GreaterEqualGreaterEqual8dropout_55/dropout/random_uniform/RandomUniform:output:0*dropout_55/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:???????????
dropout_55/dropout/CastCast#dropout_55/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:???????????
dropout_55/dropout/Mul_1Muldropout_55/dropout/Mul:z:0dropout_55/dropout/Cast:y:0*
T0*0
_output_shapes
:???????????
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_47/Conv2DConv2Ddropout_55/dropout/Mul_1:z:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_47/MaxPoolMaxPoolconv2d_47/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
]
dropout_56/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_56/dropout/MulMul!max_pooling2d_47/MaxPool:output:0!dropout_56/dropout/Const:output:0*
T0*0
_output_shapes
:??????????i
dropout_56/dropout/ShapeShape!max_pooling2d_47/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_56/dropout/random_uniform/RandomUniformRandomUniform!dropout_56/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0f
!dropout_56/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_56/dropout/GreaterEqualGreaterEqual8dropout_56/dropout/random_uniform/RandomUniform:output:0*dropout_56/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:???????????
dropout_56/dropout/CastCast#dropout_56/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:???????????
dropout_56/dropout/Mul_1Muldropout_56/dropout/Mul:z:0dropout_56/dropout/Cast:y:0*
T0*0
_output_shapes
:???????????
!dense_61/Tensordot/ReadVariableOpReadVariableOp*dense_61_tensordot_readvariableop_resource*
_output_shapes

:D*
dtype0a
dense_61/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_61/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       P
dense_61/Tensordot/ShapeShapeinputs_1*
T0*
_output_shapes
:b
 dense_61/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/GatherV2GatherV2!dense_61/Tensordot/Shape:output:0 dense_61/Tensordot/free:output:0)dense_61/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_61/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/GatherV2_1GatherV2!dense_61/Tensordot/Shape:output:0 dense_61/Tensordot/axes:output:0+dense_61/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_61/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_61/Tensordot/ProdProd$dense_61/Tensordot/GatherV2:output:0!dense_61/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_61/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_61/Tensordot/Prod_1Prod&dense_61/Tensordot/GatherV2_1:output:0#dense_61/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_61/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/concatConcatV2 dense_61/Tensordot/free:output:0 dense_61/Tensordot/axes:output:0'dense_61/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_61/Tensordot/stackPack dense_61/Tensordot/Prod:output:0"dense_61/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_61/Tensordot/transpose	Transposeinputs_1"dense_61/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????D?
dense_61/Tensordot/ReshapeReshape dense_61/Tensordot/transpose:y:0!dense_61/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_61/Tensordot/MatMulMatMul#dense_61/Tensordot/Reshape:output:0)dense_61/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Dd
dense_61/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Db
 dense_61/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/concat_1ConcatV2$dense_61/Tensordot/GatherV2:output:0#dense_61/Tensordot/Const_2:output:0)dense_61/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_61/TensordotReshape#dense_61/Tensordot/MatMul:product:0$dense_61/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????DD?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype0?
dense_61/BiasAddBiasAdddense_61/Tensordot:output:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????DDf
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*+
_output_shapes
:?????????DD?
!dense_63/Tensordot/ReadVariableOpReadVariableOp*dense_63_tensordot_readvariableop_resource*
_output_shapes

:H*
dtype0a
dense_63/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_63/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       P
dense_63/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:b
 dense_63/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/GatherV2GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/free:output:0)dense_63/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_63/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/GatherV2_1GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/axes:output:0+dense_63/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_63/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_63/Tensordot/ProdProd$dense_63/Tensordot/GatherV2:output:0!dense_63/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_63/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_63/Tensordot/Prod_1Prod&dense_63/Tensordot/GatherV2_1:output:0#dense_63/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_63/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/concatConcatV2 dense_63/Tensordot/free:output:0 dense_63/Tensordot/axes:output:0'dense_63/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_63/Tensordot/stackPack dense_63/Tensordot/Prod:output:0"dense_63/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_63/Tensordot/transpose	Transposeinputs_2"dense_63/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????H?
dense_63/Tensordot/ReshapeReshape dense_63/Tensordot/transpose:y:0!dense_63/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_63/Tensordot/MatMulMatMul#dense_63/Tensordot/Reshape:output:0)dense_63/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hd
dense_63/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Hb
 dense_63/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/concat_1ConcatV2$dense_63/Tensordot/GatherV2:output:0#dense_63/Tensordot/Const_2:output:0)dense_63/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_63/TensordotReshape#dense_63/Tensordot/MatMul:product:0$dense_63/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????HH?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
dense_63/BiasAddBiasAdddense_63/Tensordot:output:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????HHf
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*+
_output_shapes
:?????????HHa
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_28/ReshapeReshapedropout_56/dropout/Mul_1:z:0flatten_28/Const:output:0*
T0*(
_output_shapes
:??????????a
flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ?
flatten_29/ReshapeReshapedense_61/Relu:activations:0flatten_29/Const:output:0*
T0*(
_output_shapes
:??????????$a
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten_30/ReshapeReshapedense_63/Relu:activations:0flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????(?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_60/MatMulMatMulflatten_28/Reshape:output:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype0?
dense_62/MatMulMatMulflatten_29/Reshape:output:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_62/ReluReludense_62/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource* 
_output_shapes
:
?(?*
dtype0?
dense_64/MatMulMatMulflatten_30/Reshape:output:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*(
_output_shapes
:??????????]
dropout_59/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_59/dropout/MulMuldense_64/Relu:activations:0!dropout_59/dropout/Const:output:0*
T0*(
_output_shapes
:??????????c
dropout_59/dropout/ShapeShapedense_64/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_59/dropout/random_uniform/RandomUniformRandomUniform!dropout_59/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_59/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_59/dropout/GreaterEqualGreaterEqual8dropout_59/dropout/random_uniform/RandomUniform:output:0*dropout_59/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_59/dropout/CastCast#dropout_59/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_59/dropout/Mul_1Muldropout_59/dropout/Mul:z:0dropout_59/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????]
dropout_58/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_58/dropout/MulMuldense_62/Relu:activations:0!dropout_58/dropout/Const:output:0*
T0*(
_output_shapes
:??????????c
dropout_58/dropout/ShapeShapedense_62/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_58/dropout/random_uniform/RandomUniformRandomUniform!dropout_58/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_58/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_58/dropout/GreaterEqualGreaterEqual8dropout_58/dropout/random_uniform/RandomUniform:output:0*dropout_58/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_58/dropout/CastCast#dropout_58/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_58/dropout/Mul_1Muldropout_58/dropout/Mul:z:0dropout_58/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????]
dropout_57/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_57/dropout/MulMuldense_60/Relu:activations:0!dropout_57/dropout/Const:output:0*
T0*(
_output_shapes
:??????????c
dropout_57/dropout/ShapeShapedense_60/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_57/dropout/random_uniform/RandomUniformRandomUniform!dropout_57/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_57/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_57/dropout/GreaterEqualGreaterEqual8dropout_57/dropout/random_uniform/RandomUniform:output:0*dropout_57/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_57/dropout/CastCast#dropout_57/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_57/dropout/Mul_1Muldropout_57/dropout/Mul:z:0dropout_57/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_7/concatConcatV2dropout_59/dropout/Mul_1:z:0dropout_58/dropout/Mul_1:z:0dropout_57/dropout/Mul_1:z:0"concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????a
flatten_31/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_31/ReshapeReshapeconcatenate_7/concat:output:0flatten_31/Const:output:0*
T0*(
_output_shapes
:???????????
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_65/MatMulMatMulflatten_31/Reshape:output:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_66/MatMulMatMuldense_65/Relu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_66/ReluReludense_66/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_67/MatMulMatMuldense_66/Relu:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_67/SoftmaxSoftmaxdense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_67/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp"^dense_61/Tensordot/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp"^dense_63/Tensordot/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2F
!dense_61/Tensordot/ReadVariableOp!dense_61/Tensordot/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2F
!dense_63/Tensordot/ReadVariableOp!dense_63/Tensordot/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????00
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????D
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
inputs/2
?
a
E__inference_flatten_31_layer_call_and_return_conditional_losses_89975

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_56_layer_call_and_return_conditional_losses_89763

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_7_layer_call_fn_89057
inputs_0
inputs_1
inputs_2!
unknown:@
	unknown_0:@$
	unknown_1:@?
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:D
	unknown_8:D
	unknown_9:H

unknown_10:H

unknown_11:
??

unknown_12:	?

unknown_13:
?$?

unknown_14:	?

unknown_15:
?(?

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?

unknown_21:	?

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_7_layer_call_and_return_conditional_losses_88170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????00
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????D
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
inputs/2
?
?
(__inference_dense_63_layer_call_fn_89677

inputs
unknown:H
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????HH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_87998s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????HH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
D__inference_conv2d_47_layer_call_and_return_conditional_losses_89648

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_58_layer_call_fn_89900

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_88091a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_56_layer_call_fn_89753

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_56_layer_call_and_return_conditional_losses_87928i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87791

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_54_layer_call_and_return_conditional_losses_88498

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????

?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????

?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????

?x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????

?r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????

?b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????

?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????

?:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?

d
E__inference_dropout_54_layer_call_and_return_conditional_losses_89561

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????

?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????

?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????

?x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????

?r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????

?b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????

?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????

?:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?
c
E__inference_dropout_59_layer_call_and_return_conditional_losses_89883

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_89663

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_30_layer_call_and_return_conditional_losses_89786

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????HH:S O
+
_output_shapes
:?????????HH
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_47_layer_call_fn_89653

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87803?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_58_layer_call_and_return_conditional_losses_89922

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_67_layer_call_fn_90024

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_88163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_63_layer_call_and_return_conditional_losses_87998

inputs3
!tensordot_readvariableop_resource:H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:H*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????H?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:HY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????HHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????HHT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????HHe
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????HHz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_89514

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_87758
	convinput
landmarksinput
hoginputJ
0model_7_conv2d_44_conv2d_readvariableop_resource:@?
1model_7_conv2d_44_biasadd_readvariableop_resource:@K
0model_7_conv2d_45_conv2d_readvariableop_resource:@?@
1model_7_conv2d_45_biasadd_readvariableop_resource:	?L
0model_7_conv2d_46_conv2d_readvariableop_resource:??@
1model_7_conv2d_46_biasadd_readvariableop_resource:	?L
0model_7_conv2d_47_conv2d_readvariableop_resource:??@
1model_7_conv2d_47_biasadd_readvariableop_resource:	?D
2model_7_dense_61_tensordot_readvariableop_resource:D>
0model_7_dense_61_biasadd_readvariableop_resource:DD
2model_7_dense_63_tensordot_readvariableop_resource:H>
0model_7_dense_63_biasadd_readvariableop_resource:HC
/model_7_dense_60_matmul_readvariableop_resource:
???
0model_7_dense_60_biasadd_readvariableop_resource:	?C
/model_7_dense_62_matmul_readvariableop_resource:
?$??
0model_7_dense_62_biasadd_readvariableop_resource:	?C
/model_7_dense_64_matmul_readvariableop_resource:
?(??
0model_7_dense_64_biasadd_readvariableop_resource:	?C
/model_7_dense_65_matmul_readvariableop_resource:
???
0model_7_dense_65_biasadd_readvariableop_resource:	?C
/model_7_dense_66_matmul_readvariableop_resource:
???
0model_7_dense_66_biasadd_readvariableop_resource:	?B
/model_7_dense_67_matmul_readvariableop_resource:	?>
0model_7_dense_67_biasadd_readvariableop_resource:
identity??(model_7/conv2d_44/BiasAdd/ReadVariableOp?'model_7/conv2d_44/Conv2D/ReadVariableOp?(model_7/conv2d_45/BiasAdd/ReadVariableOp?'model_7/conv2d_45/Conv2D/ReadVariableOp?(model_7/conv2d_46/BiasAdd/ReadVariableOp?'model_7/conv2d_46/Conv2D/ReadVariableOp?(model_7/conv2d_47/BiasAdd/ReadVariableOp?'model_7/conv2d_47/Conv2D/ReadVariableOp?'model_7/dense_60/BiasAdd/ReadVariableOp?&model_7/dense_60/MatMul/ReadVariableOp?'model_7/dense_61/BiasAdd/ReadVariableOp?)model_7/dense_61/Tensordot/ReadVariableOp?'model_7/dense_62/BiasAdd/ReadVariableOp?&model_7/dense_62/MatMul/ReadVariableOp?'model_7/dense_63/BiasAdd/ReadVariableOp?)model_7/dense_63/Tensordot/ReadVariableOp?'model_7/dense_64/BiasAdd/ReadVariableOp?&model_7/dense_64/MatMul/ReadVariableOp?'model_7/dense_65/BiasAdd/ReadVariableOp?&model_7/dense_65/MatMul/ReadVariableOp?'model_7/dense_66/BiasAdd/ReadVariableOp?&model_7/dense_66/MatMul/ReadVariableOp?'model_7/dense_67/BiasAdd/ReadVariableOp?&model_7/dense_67/MatMul/ReadVariableOp?
'model_7/conv2d_44/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model_7/conv2d_44/Conv2DConv2D	convinput/model_7/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@*
paddingVALID*
strides
?
(model_7/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_7/conv2d_44/BiasAddBiasAdd!model_7/conv2d_44/Conv2D:output:00model_7/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@|
model_7/conv2d_44/ReluRelu"model_7/conv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:?????????,,@?
 model_7/max_pooling2d_44/MaxPoolMaxPool$model_7/conv2d_44/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
'model_7/conv2d_45/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_45_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
model_7/conv2d_45/Conv2DConv2D)model_7/max_pooling2d_44/MaxPool:output:0/model_7/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
(model_7/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/conv2d_45/BiasAddBiasAdd!model_7/conv2d_45/Conv2D:output:00model_7/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????}
model_7/conv2d_45/ReluRelu"model_7/conv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
 model_7/max_pooling2d_45/MaxPoolMaxPool$model_7/conv2d_45/Relu:activations:0*0
_output_shapes
:?????????

?*
ksize
*
paddingVALID*
strides
?
model_7/dropout_54/IdentityIdentity)model_7/max_pooling2d_45/MaxPool:output:0*
T0*0
_output_shapes
:?????????

??
'model_7/conv2d_46/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
model_7/conv2d_46/Conv2DConv2D$model_7/dropout_54/Identity:output:0/model_7/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
(model_7/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/conv2d_46/BiasAddBiasAdd!model_7/conv2d_46/Conv2D:output:00model_7/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????}
model_7/conv2d_46/ReluRelu"model_7/conv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
 model_7/max_pooling2d_46/MaxPoolMaxPool$model_7/conv2d_46/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
model_7/dropout_55/IdentityIdentity)model_7/max_pooling2d_46/MaxPool:output:0*
T0*0
_output_shapes
:???????????
'model_7/conv2d_47/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
model_7/conv2d_47/Conv2DConv2D$model_7/dropout_55/Identity:output:0/model_7/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
(model_7/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/conv2d_47/BiasAddBiasAdd!model_7/conv2d_47/Conv2D:output:00model_7/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????}
model_7/conv2d_47/ReluRelu"model_7/conv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
 model_7/max_pooling2d_47/MaxPoolMaxPool$model_7/conv2d_47/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
model_7/dropout_56/IdentityIdentity)model_7/max_pooling2d_47/MaxPool:output:0*
T0*0
_output_shapes
:???????????
)model_7/dense_61/Tensordot/ReadVariableOpReadVariableOp2model_7_dense_61_tensordot_readvariableop_resource*
_output_shapes

:D*
dtype0i
model_7/dense_61/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
model_7/dense_61/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ^
 model_7/dense_61/Tensordot/ShapeShapelandmarksinput*
T0*
_output_shapes
:j
(model_7/dense_61/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_7/dense_61/Tensordot/GatherV2GatherV2)model_7/dense_61/Tensordot/Shape:output:0(model_7/dense_61/Tensordot/free:output:01model_7/dense_61/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_7/dense_61/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%model_7/dense_61/Tensordot/GatherV2_1GatherV2)model_7/dense_61/Tensordot/Shape:output:0(model_7/dense_61/Tensordot/axes:output:03model_7/dense_61/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_7/dense_61/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
model_7/dense_61/Tensordot/ProdProd,model_7/dense_61/Tensordot/GatherV2:output:0)model_7/dense_61/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_7/dense_61/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
!model_7/dense_61/Tensordot/Prod_1Prod.model_7/dense_61/Tensordot/GatherV2_1:output:0+model_7/dense_61/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_7/dense_61/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!model_7/dense_61/Tensordot/concatConcatV2(model_7/dense_61/Tensordot/free:output:0(model_7/dense_61/Tensordot/axes:output:0/model_7/dense_61/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 model_7/dense_61/Tensordot/stackPack(model_7/dense_61/Tensordot/Prod:output:0*model_7/dense_61/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
$model_7/dense_61/Tensordot/transpose	Transposelandmarksinput*model_7/dense_61/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????D?
"model_7/dense_61/Tensordot/ReshapeReshape(model_7/dense_61/Tensordot/transpose:y:0)model_7/dense_61/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
!model_7/dense_61/Tensordot/MatMulMatMul+model_7/dense_61/Tensordot/Reshape:output:01model_7/dense_61/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Dl
"model_7/dense_61/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Dj
(model_7/dense_61/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_7/dense_61/Tensordot/concat_1ConcatV2,model_7/dense_61/Tensordot/GatherV2:output:0+model_7/dense_61/Tensordot/Const_2:output:01model_7/dense_61/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
model_7/dense_61/TensordotReshape+model_7/dense_61/Tensordot/MatMul:product:0,model_7/dense_61/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????DD?
'model_7/dense_61/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_61_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype0?
model_7/dense_61/BiasAddBiasAdd#model_7/dense_61/Tensordot:output:0/model_7/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????DDv
model_7/dense_61/ReluRelu!model_7/dense_61/BiasAdd:output:0*
T0*+
_output_shapes
:?????????DD?
)model_7/dense_63/Tensordot/ReadVariableOpReadVariableOp2model_7_dense_63_tensordot_readvariableop_resource*
_output_shapes

:H*
dtype0i
model_7/dense_63/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
model_7/dense_63/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       X
 model_7/dense_63/Tensordot/ShapeShapehoginput*
T0*
_output_shapes
:j
(model_7/dense_63/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_7/dense_63/Tensordot/GatherV2GatherV2)model_7/dense_63/Tensordot/Shape:output:0(model_7/dense_63/Tensordot/free:output:01model_7/dense_63/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_7/dense_63/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%model_7/dense_63/Tensordot/GatherV2_1GatherV2)model_7/dense_63/Tensordot/Shape:output:0(model_7/dense_63/Tensordot/axes:output:03model_7/dense_63/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_7/dense_63/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
model_7/dense_63/Tensordot/ProdProd,model_7/dense_63/Tensordot/GatherV2:output:0)model_7/dense_63/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_7/dense_63/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
!model_7/dense_63/Tensordot/Prod_1Prod.model_7/dense_63/Tensordot/GatherV2_1:output:0+model_7/dense_63/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_7/dense_63/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!model_7/dense_63/Tensordot/concatConcatV2(model_7/dense_63/Tensordot/free:output:0(model_7/dense_63/Tensordot/axes:output:0/model_7/dense_63/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 model_7/dense_63/Tensordot/stackPack(model_7/dense_63/Tensordot/Prod:output:0*model_7/dense_63/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
$model_7/dense_63/Tensordot/transpose	Transposehoginput*model_7/dense_63/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????H?
"model_7/dense_63/Tensordot/ReshapeReshape(model_7/dense_63/Tensordot/transpose:y:0)model_7/dense_63/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
!model_7/dense_63/Tensordot/MatMulMatMul+model_7/dense_63/Tensordot/Reshape:output:01model_7/dense_63/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hl
"model_7/dense_63/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Hj
(model_7/dense_63/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_7/dense_63/Tensordot/concat_1ConcatV2,model_7/dense_63/Tensordot/GatherV2:output:0+model_7/dense_63/Tensordot/Const_2:output:01model_7/dense_63/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
model_7/dense_63/TensordotReshape+model_7/dense_63/Tensordot/MatMul:product:0,model_7/dense_63/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????HH?
'model_7/dense_63/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_63_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
model_7/dense_63/BiasAddBiasAdd#model_7/dense_63/Tensordot:output:0/model_7/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????HHv
model_7/dense_63/ReluRelu!model_7/dense_63/BiasAdd:output:0*
T0*+
_output_shapes
:?????????HHi
model_7/flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
model_7/flatten_28/ReshapeReshape$model_7/dropout_56/Identity:output:0!model_7/flatten_28/Const:output:0*
T0*(
_output_shapes
:??????????i
model_7/flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ?
model_7/flatten_29/ReshapeReshape#model_7/dense_61/Relu:activations:0!model_7/flatten_29/Const:output:0*
T0*(
_output_shapes
:??????????$i
model_7/flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
model_7/flatten_30/ReshapeReshape#model_7/dense_63/Relu:activations:0!model_7/flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????(?
&model_7/dense_60/MatMul/ReadVariableOpReadVariableOp/model_7_dense_60_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_7/dense_60/MatMulMatMul#model_7/flatten_28/Reshape:output:0.model_7/dense_60/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'model_7/dense_60/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_60_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/dense_60/BiasAddBiasAdd!model_7/dense_60/MatMul:product:0/model_7/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
model_7/dense_60/ReluRelu!model_7/dense_60/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
&model_7/dense_62/MatMul/ReadVariableOpReadVariableOp/model_7_dense_62_matmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype0?
model_7/dense_62/MatMulMatMul#model_7/flatten_29/Reshape:output:0.model_7/dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'model_7/dense_62/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/dense_62/BiasAddBiasAdd!model_7/dense_62/MatMul:product:0/model_7/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
model_7/dense_62/ReluRelu!model_7/dense_62/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
&model_7/dense_64/MatMul/ReadVariableOpReadVariableOp/model_7_dense_64_matmul_readvariableop_resource* 
_output_shapes
:
?(?*
dtype0?
model_7/dense_64/MatMulMatMul#model_7/flatten_30/Reshape:output:0.model_7/dense_64/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'model_7/dense_64/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_64_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/dense_64/BiasAddBiasAdd!model_7/dense_64/MatMul:product:0/model_7/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
model_7/dense_64/ReluRelu!model_7/dense_64/BiasAdd:output:0*
T0*(
_output_shapes
:??????????
model_7/dropout_59/IdentityIdentity#model_7/dense_64/Relu:activations:0*
T0*(
_output_shapes
:??????????
model_7/dropout_58/IdentityIdentity#model_7/dense_62/Relu:activations:0*
T0*(
_output_shapes
:??????????
model_7/dropout_57/IdentityIdentity#model_7/dense_60/Relu:activations:0*
T0*(
_output_shapes
:??????????c
!model_7/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_7/concatenate_7/concatConcatV2$model_7/dropout_59/Identity:output:0$model_7/dropout_58/Identity:output:0$model_7/dropout_57/Identity:output:0*model_7/concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????i
model_7/flatten_31/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
model_7/flatten_31/ReshapeReshape%model_7/concatenate_7/concat:output:0!model_7/flatten_31/Const:output:0*
T0*(
_output_shapes
:???????????
&model_7/dense_65/MatMul/ReadVariableOpReadVariableOp/model_7_dense_65_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_7/dense_65/MatMulMatMul#model_7/flatten_31/Reshape:output:0.model_7/dense_65/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'model_7/dense_65/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_65_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/dense_65/BiasAddBiasAdd!model_7/dense_65/MatMul:product:0/model_7/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
model_7/dense_65/ReluRelu!model_7/dense_65/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
&model_7/dense_66/MatMul/ReadVariableOpReadVariableOp/model_7_dense_66_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_7/dense_66/MatMulMatMul#model_7/dense_65/Relu:activations:0.model_7/dense_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'model_7/dense_66/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_66_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_7/dense_66/BiasAddBiasAdd!model_7/dense_66/MatMul:product:0/model_7/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
model_7/dense_66/ReluRelu!model_7/dense_66/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
&model_7/dense_67/MatMul/ReadVariableOpReadVariableOp/model_7_dense_67_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_7/dense_67/MatMulMatMul#model_7/dense_66/Relu:activations:0.model_7/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'model_7/dense_67/BiasAdd/ReadVariableOpReadVariableOp0model_7_dense_67_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_7/dense_67/BiasAddBiasAdd!model_7/dense_67/MatMul:product:0/model_7/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_7/dense_67/SoftmaxSoftmax!model_7/dense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"model_7/dense_67/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^model_7/conv2d_44/BiasAdd/ReadVariableOp(^model_7/conv2d_44/Conv2D/ReadVariableOp)^model_7/conv2d_45/BiasAdd/ReadVariableOp(^model_7/conv2d_45/Conv2D/ReadVariableOp)^model_7/conv2d_46/BiasAdd/ReadVariableOp(^model_7/conv2d_46/Conv2D/ReadVariableOp)^model_7/conv2d_47/BiasAdd/ReadVariableOp(^model_7/conv2d_47/Conv2D/ReadVariableOp(^model_7/dense_60/BiasAdd/ReadVariableOp'^model_7/dense_60/MatMul/ReadVariableOp(^model_7/dense_61/BiasAdd/ReadVariableOp*^model_7/dense_61/Tensordot/ReadVariableOp(^model_7/dense_62/BiasAdd/ReadVariableOp'^model_7/dense_62/MatMul/ReadVariableOp(^model_7/dense_63/BiasAdd/ReadVariableOp*^model_7/dense_63/Tensordot/ReadVariableOp(^model_7/dense_64/BiasAdd/ReadVariableOp'^model_7/dense_64/MatMul/ReadVariableOp(^model_7/dense_65/BiasAdd/ReadVariableOp'^model_7/dense_65/MatMul/ReadVariableOp(^model_7/dense_66/BiasAdd/ReadVariableOp'^model_7/dense_66/MatMul/ReadVariableOp(^model_7/dense_67/BiasAdd/ReadVariableOp'^model_7/dense_67/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_7/conv2d_44/BiasAdd/ReadVariableOp(model_7/conv2d_44/BiasAdd/ReadVariableOp2R
'model_7/conv2d_44/Conv2D/ReadVariableOp'model_7/conv2d_44/Conv2D/ReadVariableOp2T
(model_7/conv2d_45/BiasAdd/ReadVariableOp(model_7/conv2d_45/BiasAdd/ReadVariableOp2R
'model_7/conv2d_45/Conv2D/ReadVariableOp'model_7/conv2d_45/Conv2D/ReadVariableOp2T
(model_7/conv2d_46/BiasAdd/ReadVariableOp(model_7/conv2d_46/BiasAdd/ReadVariableOp2R
'model_7/conv2d_46/Conv2D/ReadVariableOp'model_7/conv2d_46/Conv2D/ReadVariableOp2T
(model_7/conv2d_47/BiasAdd/ReadVariableOp(model_7/conv2d_47/BiasAdd/ReadVariableOp2R
'model_7/conv2d_47/Conv2D/ReadVariableOp'model_7/conv2d_47/Conv2D/ReadVariableOp2R
'model_7/dense_60/BiasAdd/ReadVariableOp'model_7/dense_60/BiasAdd/ReadVariableOp2P
&model_7/dense_60/MatMul/ReadVariableOp&model_7/dense_60/MatMul/ReadVariableOp2R
'model_7/dense_61/BiasAdd/ReadVariableOp'model_7/dense_61/BiasAdd/ReadVariableOp2V
)model_7/dense_61/Tensordot/ReadVariableOp)model_7/dense_61/Tensordot/ReadVariableOp2R
'model_7/dense_62/BiasAdd/ReadVariableOp'model_7/dense_62/BiasAdd/ReadVariableOp2P
&model_7/dense_62/MatMul/ReadVariableOp&model_7/dense_62/MatMul/ReadVariableOp2R
'model_7/dense_63/BiasAdd/ReadVariableOp'model_7/dense_63/BiasAdd/ReadVariableOp2V
)model_7/dense_63/Tensordot/ReadVariableOp)model_7/dense_63/Tensordot/ReadVariableOp2R
'model_7/dense_64/BiasAdd/ReadVariableOp'model_7/dense_64/BiasAdd/ReadVariableOp2P
&model_7/dense_64/MatMul/ReadVariableOp&model_7/dense_64/MatMul/ReadVariableOp2R
'model_7/dense_65/BiasAdd/ReadVariableOp'model_7/dense_65/BiasAdd/ReadVariableOp2P
&model_7/dense_65/MatMul/ReadVariableOp&model_7/dense_65/MatMul/ReadVariableOp2R
'model_7/dense_66/BiasAdd/ReadVariableOp'model_7/dense_66/BiasAdd/ReadVariableOp2P
&model_7/dense_66/MatMul/ReadVariableOp&model_7/dense_66/MatMul/ReadVariableOp2R
'model_7/dense_67/BiasAdd/ReadVariableOp'model_7/dense_67/BiasAdd/ReadVariableOp2P
&model_7/dense_67/MatMul/ReadVariableOp&model_7/dense_67/MatMul/ReadVariableOp:Z V
/
_output_shapes
:?????????00
#
_user_specified_name	ConvInput:[W
+
_output_shapes
:?????????D
(
_user_specified_nameLandmarksInput:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
HogInput
?
c
*__inference_dropout_57_layer_call_fn_89932

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_88285p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_46_layer_call_fn_89570

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_87881x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?j
?

B__inference_model_7_layer_call_and_return_conditional_losses_88170

inputs
inputs_1
inputs_2)
conv2d_44_87829:@
conv2d_44_87831:@*
conv2d_45_87852:@?
conv2d_45_87854:	?+
conv2d_46_87882:??
conv2d_46_87884:	?+
conv2d_47_87912:??
conv2d_47_87914:	? 
dense_61_87962:D
dense_61_87964:D 
dense_63_87999:H
dense_63_88001:H"
dense_60_88040:
??
dense_60_88042:	?"
dense_62_88057:
?$?
dense_62_88059:	?"
dense_64_88074:
?(?
dense_64_88076:	?"
dense_65_88130:
??
dense_65_88132:	?"
dense_66_88147:
??
dense_66_88149:	?!
dense_67_88164:	?
dense_67_88166:
identity??!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_44_87829conv2d_44_87831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,,@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_87828?
 max_pooling2d_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87838?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_44/PartitionedCall:output:0conv2d_45_87852conv2d_45_87854*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_87851?
 max_pooling2d_45/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87861?
dropout_54/PartitionedCallPartitionedCall)max_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_54_layer_call_and_return_conditional_losses_87868?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall#dropout_54/PartitionedCall:output:0conv2d_46_87882conv2d_46_87884*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_87881?
 max_pooling2d_46/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87891?
dropout_55/PartitionedCallPartitionedCall)max_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_55_layer_call_and_return_conditional_losses_87898?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall#dropout_55/PartitionedCall:output:0conv2d_47_87912conv2d_47_87914*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_87911?
 max_pooling2d_47/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87921?
dropout_56/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_56_layer_call_and_return_conditional_losses_87928?
 dense_61/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_61_87962dense_61_87964*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????DD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_61_layer_call_and_return_conditional_losses_87961?
 dense_63/StatefulPartitionedCallStatefulPartitionedCallinputs_2dense_63_87999dense_63_88001*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????HH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_87998?
flatten_28/PartitionedCallPartitionedCall#dropout_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_88010?
flatten_29/PartitionedCallPartitionedCall)dense_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_29_layer_call_and_return_conditional_losses_88018?
flatten_30/PartitionedCallPartitionedCall)dense_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_30_layer_call_and_return_conditional_losses_88026?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_60_88040dense_60_88042*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_60_layer_call_and_return_conditional_losses_88039?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_62_88057dense_62_88059*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_62_layer_call_and_return_conditional_losses_88056?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_64_88074dense_64_88076*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_88073?
dropout_59/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_59_layer_call_and_return_conditional_losses_88084?
dropout_58/PartitionedCallPartitionedCall)dense_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_88091?
dropout_57/PartitionedCallPartitionedCall)dense_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_88098?
concatenate_7/PartitionedCallPartitionedCall#dropout_59/PartitionedCall:output:0#dropout_58/PartitionedCall:output:0#dropout_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_88108?
flatten_31/PartitionedCallPartitionedCall&concatenate_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_31_layer_call_and_return_conditional_losses_88116?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall#flatten_31/PartitionedCall:output:0dense_65_88130dense_65_88132*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_88129?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_88147dense_66_88149*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_88146?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_88164dense_67_88166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_88163x
IdentityIdentity)dense_67/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????D
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
D__inference_conv2d_46_layer_call_and_return_conditional_losses_87881

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?s
?
B__inference_model_7_layer_call_and_return_conditional_losses_88939
	convinput
landmarksinput
hoginput)
conv2d_44_88863:@
conv2d_44_88865:@*
conv2d_45_88869:@?
conv2d_45_88871:	?+
conv2d_46_88876:??
conv2d_46_88878:	?+
conv2d_47_88883:??
conv2d_47_88885:	? 
dense_61_88890:D
dense_61_88892:D 
dense_63_88895:H
dense_63_88897:H"
dense_60_88903:
??
dense_60_88905:	?"
dense_62_88908:
?$?
dense_62_88910:	?"
dense_64_88913:
?(?
dense_64_88915:	?"
dense_65_88923:
??
dense_65_88925:	?"
dense_66_88928:
??
dense_66_88930:	?!
dense_67_88933:	?
dense_67_88935:
identity??!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall?"dropout_54/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?"dropout_56/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?"dropout_58/StatefulPartitionedCall?"dropout_59/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall	convinputconv2d_44_88863conv2d_44_88865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,,@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_87828?
 max_pooling2d_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87838?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_44/PartitionedCall:output:0conv2d_45_88869conv2d_45_88871*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_87851?
 max_pooling2d_45/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87861?
"dropout_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_54_layer_call_and_return_conditional_losses_88498?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall+dropout_54/StatefulPartitionedCall:output:0conv2d_46_88876conv2d_46_88878*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_87881?
 max_pooling2d_46/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87891?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0#^dropout_54/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_55_layer_call_and_return_conditional_losses_88460?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall+dropout_55/StatefulPartitionedCall:output:0conv2d_47_88883conv2d_47_88885*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_87911?
 max_pooling2d_47/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87921?
"dropout_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0#^dropout_55/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_56_layer_call_and_return_conditional_losses_88422?
 dense_61/StatefulPartitionedCallStatefulPartitionedCalllandmarksinputdense_61_88890dense_61_88892*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????DD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_61_layer_call_and_return_conditional_losses_87961?
 dense_63/StatefulPartitionedCallStatefulPartitionedCallhoginputdense_63_88895dense_63_88897*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????HH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_87998?
flatten_28/PartitionedCallPartitionedCall+dropout_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_88010?
flatten_29/PartitionedCallPartitionedCall)dense_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_29_layer_call_and_return_conditional_losses_88018?
flatten_30/PartitionedCallPartitionedCall)dense_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_30_layer_call_and_return_conditional_losses_88026?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_60_88903dense_60_88905*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_60_layer_call_and_return_conditional_losses_88039?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_62_88908dense_62_88910*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_62_layer_call_and_return_conditional_losses_88056?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_64_88913dense_64_88915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_88073?
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0#^dropout_56/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_59_layer_call_and_return_conditional_losses_88331?
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall)dense_62/StatefulPartitionedCall:output:0#^dropout_59/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_88308?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0#^dropout_58/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_88285?
concatenate_7/PartitionedCallPartitionedCall+dropout_59/StatefulPartitionedCall:output:0+dropout_58/StatefulPartitionedCall:output:0+dropout_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_88108?
flatten_31/PartitionedCallPartitionedCall&concatenate_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_31_layer_call_and_return_conditional_losses_88116?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall#flatten_31/PartitionedCall:output:0dense_65_88923dense_65_88925*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_88129?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_88928dense_66_88930*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_88146?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_88933dense_67_88935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_88163x
IdentityIdentity)dense_67/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall#^dropout_54/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall#^dropout_56/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2H
"dropout_54/StatefulPartitionedCall"dropout_54/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall2H
"dropout_56/StatefulPartitionedCall"dropout_56/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall:Z V
/
_output_shapes
:?????????00
#
_user_specified_name	ConvInput:[W
+
_output_shapes
:?????????D
(
_user_specified_nameLandmarksInput:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
HogInput
?
?
H__inference_concatenate_7_layer_call_and_return_conditional_losses_89964
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?

d
E__inference_dropout_55_layer_call_and_return_conditional_losses_88460

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
B__inference_model_7_layer_call_and_return_conditional_losses_89262
inputs_0
inputs_1
inputs_2B
(conv2d_44_conv2d_readvariableop_resource:@7
)conv2d_44_biasadd_readvariableop_resource:@C
(conv2d_45_conv2d_readvariableop_resource:@?8
)conv2d_45_biasadd_readvariableop_resource:	?D
(conv2d_46_conv2d_readvariableop_resource:??8
)conv2d_46_biasadd_readvariableop_resource:	?D
(conv2d_47_conv2d_readvariableop_resource:??8
)conv2d_47_biasadd_readvariableop_resource:	?<
*dense_61_tensordot_readvariableop_resource:D6
(dense_61_biasadd_readvariableop_resource:D<
*dense_63_tensordot_readvariableop_resource:H6
(dense_63_biasadd_readvariableop_resource:H;
'dense_60_matmul_readvariableop_resource:
??7
(dense_60_biasadd_readvariableop_resource:	?;
'dense_62_matmul_readvariableop_resource:
?$?7
(dense_62_biasadd_readvariableop_resource:	?;
'dense_64_matmul_readvariableop_resource:
?(?7
(dense_64_biasadd_readvariableop_resource:	?;
'dense_65_matmul_readvariableop_resource:
??7
(dense_65_biasadd_readvariableop_resource:	?;
'dense_66_matmul_readvariableop_resource:
??7
(dense_66_biasadd_readvariableop_resource:	?:
'dense_67_matmul_readvariableop_resource:	?6
(dense_67_biasadd_readvariableop_resource:
identity?? conv2d_44/BiasAdd/ReadVariableOp?conv2d_44/Conv2D/ReadVariableOp? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp? conv2d_46/BiasAdd/ReadVariableOp?conv2d_46/Conv2D/ReadVariableOp? conv2d_47/BiasAdd/ReadVariableOp?conv2d_47/Conv2D/ReadVariableOp?dense_60/BiasAdd/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/BiasAdd/ReadVariableOp?!dense_61/Tensordot/ReadVariableOp?dense_62/BiasAdd/ReadVariableOp?dense_62/MatMul/ReadVariableOp?dense_63/BiasAdd/ReadVariableOp?!dense_63/Tensordot/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?dense_65/MatMul/ReadVariableOp?dense_66/BiasAdd/ReadVariableOp?dense_66/MatMul/ReadVariableOp?dense_67/BiasAdd/ReadVariableOp?dense_67/MatMul/ReadVariableOp?
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_44/Conv2DConv2Dinputs_0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@*
paddingVALID*
strides
?
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@l
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:?????????,,@?
max_pooling2d_44/MaxPoolMaxPoolconv2d_44/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_45/Conv2DConv2D!max_pooling2d_44/MaxPool:output:0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_45/MaxPoolMaxPoolconv2d_45/Relu:activations:0*0
_output_shapes
:?????????

?*
ksize
*
paddingVALID*
strides
}
dropout_54/IdentityIdentity!max_pooling2d_45/MaxPool:output:0*
T0*0
_output_shapes
:?????????

??
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_46/Conv2DConv2Ddropout_54/Identity:output:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_46/MaxPoolMaxPoolconv2d_46/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
}
dropout_55/IdentityIdentity!max_pooling2d_46/MaxPool:output:0*
T0*0
_output_shapes
:???????????
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_47/Conv2DConv2Ddropout_55/Identity:output:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_47/MaxPoolMaxPoolconv2d_47/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
}
dropout_56/IdentityIdentity!max_pooling2d_47/MaxPool:output:0*
T0*0
_output_shapes
:???????????
!dense_61/Tensordot/ReadVariableOpReadVariableOp*dense_61_tensordot_readvariableop_resource*
_output_shapes

:D*
dtype0a
dense_61/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_61/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       P
dense_61/Tensordot/ShapeShapeinputs_1*
T0*
_output_shapes
:b
 dense_61/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/GatherV2GatherV2!dense_61/Tensordot/Shape:output:0 dense_61/Tensordot/free:output:0)dense_61/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_61/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/GatherV2_1GatherV2!dense_61/Tensordot/Shape:output:0 dense_61/Tensordot/axes:output:0+dense_61/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_61/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_61/Tensordot/ProdProd$dense_61/Tensordot/GatherV2:output:0!dense_61/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_61/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_61/Tensordot/Prod_1Prod&dense_61/Tensordot/GatherV2_1:output:0#dense_61/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_61/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/concatConcatV2 dense_61/Tensordot/free:output:0 dense_61/Tensordot/axes:output:0'dense_61/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_61/Tensordot/stackPack dense_61/Tensordot/Prod:output:0"dense_61/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_61/Tensordot/transpose	Transposeinputs_1"dense_61/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????D?
dense_61/Tensordot/ReshapeReshape dense_61/Tensordot/transpose:y:0!dense_61/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_61/Tensordot/MatMulMatMul#dense_61/Tensordot/Reshape:output:0)dense_61/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Dd
dense_61/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Db
 dense_61/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_61/Tensordot/concat_1ConcatV2$dense_61/Tensordot/GatherV2:output:0#dense_61/Tensordot/Const_2:output:0)dense_61/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_61/TensordotReshape#dense_61/Tensordot/MatMul:product:0$dense_61/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????DD?
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype0?
dense_61/BiasAddBiasAdddense_61/Tensordot:output:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????DDf
dense_61/ReluReludense_61/BiasAdd:output:0*
T0*+
_output_shapes
:?????????DD?
!dense_63/Tensordot/ReadVariableOpReadVariableOp*dense_63_tensordot_readvariableop_resource*
_output_shapes

:H*
dtype0a
dense_63/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_63/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       P
dense_63/Tensordot/ShapeShapeinputs_2*
T0*
_output_shapes
:b
 dense_63/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/GatherV2GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/free:output:0)dense_63/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_63/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/GatherV2_1GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/axes:output:0+dense_63/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_63/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_63/Tensordot/ProdProd$dense_63/Tensordot/GatherV2:output:0!dense_63/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_63/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_63/Tensordot/Prod_1Prod&dense_63/Tensordot/GatherV2_1:output:0#dense_63/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_63/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/concatConcatV2 dense_63/Tensordot/free:output:0 dense_63/Tensordot/axes:output:0'dense_63/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_63/Tensordot/stackPack dense_63/Tensordot/Prod:output:0"dense_63/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_63/Tensordot/transpose	Transposeinputs_2"dense_63/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????H?
dense_63/Tensordot/ReshapeReshape dense_63/Tensordot/transpose:y:0!dense_63/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_63/Tensordot/MatMulMatMul#dense_63/Tensordot/Reshape:output:0)dense_63/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hd
dense_63/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Hb
 dense_63/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_63/Tensordot/concat_1ConcatV2$dense_63/Tensordot/GatherV2:output:0#dense_63/Tensordot/Const_2:output:0)dense_63/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_63/TensordotReshape#dense_63/Tensordot/MatMul:product:0$dense_63/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????HH?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
dense_63/BiasAddBiasAdddense_63/Tensordot:output:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????HHf
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*+
_output_shapes
:?????????HHa
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_28/ReshapeReshapedropout_56/Identity:output:0flatten_28/Const:output:0*
T0*(
_output_shapes
:??????????a
flatten_29/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ?
flatten_29/ReshapeReshapedense_61/Relu:activations:0flatten_29/Const:output:0*
T0*(
_output_shapes
:??????????$a
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten_30/ReshapeReshapedense_63/Relu:activations:0flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????(?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_60/MatMulMatMulflatten_28/Reshape:output:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_60/ReluReludense_60/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource* 
_output_shapes
:
?$?*
dtype0?
dense_62/MatMulMatMulflatten_29/Reshape:output:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_62/ReluReludense_62/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource* 
_output_shapes
:
?(?*
dtype0?
dense_64/MatMulMatMulflatten_30/Reshape:output:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
dropout_59/IdentityIdentitydense_64/Relu:activations:0*
T0*(
_output_shapes
:??????????o
dropout_58/IdentityIdentitydense_62/Relu:activations:0*
T0*(
_output_shapes
:??????????o
dropout_57/IdentityIdentitydense_60/Relu:activations:0*
T0*(
_output_shapes
:??????????[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_7/concatConcatV2dropout_59/Identity:output:0dropout_58/Identity:output:0dropout_57/Identity:output:0"concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????a
flatten_31/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_31/ReshapeReshapeconcatenate_7/concat:output:0flatten_31/Const:output:0*
T0*(
_output_shapes
:???????????
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_65/MatMulMatMulflatten_31/Reshape:output:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_66/MatMulMatMuldense_65/Relu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_66/ReluReludense_66/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_67/MatMulMatMuldense_66/Relu:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_67/SoftmaxSoftmaxdense_67/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_67/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp"^dense_61/Tensordot/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp"^dense_63/Tensordot/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2F
!dense_61/Tensordot/ReadVariableOp!dense_61/Tensordot/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2F
!dense_63/Tensordot/ReadVariableOp!dense_63/Tensordot/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????00
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????D
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
inputs/2
?
g
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87779

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_87828

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????,,@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????,,@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
a
E__inference_flatten_31_layer_call_and_return_conditional_losses_88116

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_89529

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_47_layer_call_and_return_conditional_losses_87911

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_58_layer_call_and_return_conditional_losses_88308

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_56_layer_call_and_return_conditional_losses_87928

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_64_layer_call_fn_89817

inputs
unknown:
?(?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_88073p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
(__inference_dense_62_layer_call_fn_89837

inputs
unknown:
?$?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_62_layer_call_and_return_conditional_losses_88056p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_87851

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
*__inference_dropout_59_layer_call_fn_89878

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_59_layer_call_and_return_conditional_losses_88331p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87838

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,,@:W S
/
_output_shapes
:?????????,,@
 
_user_specified_nameinputs
?
a
E__inference_flatten_28_layer_call_and_return_conditional_losses_88010

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_60_layer_call_and_return_conditional_losses_89868

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_28_layer_call_fn_89802

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_88010a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_89596

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_46_layer_call_and_return_conditional_losses_89581

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????

?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?
c
E__inference_dropout_55_layer_call_and_return_conditional_losses_89616

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_54_layer_call_and_return_conditional_losses_89549

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????

?d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????

?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????

?:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_45_layer_call_fn_89519

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87779?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_54_layer_call_and_return_conditional_losses_87868

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????

?d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????

?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????

?:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87767

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_44_layer_call_fn_89463

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,,@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_87828w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????,,@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
Л
?!
__inference__traced_save_90303
file_prefix/
+savev2_conv2d_44_kernel_read_readvariableop-
)savev2_conv2d_44_bias_read_readvariableop/
+savev2_conv2d_45_kernel_read_readvariableop-
)savev2_conv2d_45_bias_read_readvariableop/
+savev2_conv2d_46_kernel_read_readvariableop-
)savev2_conv2d_46_bias_read_readvariableop/
+savev2_conv2d_47_kernel_read_readvariableop-
)savev2_conv2d_47_bias_read_readvariableop.
*savev2_dense_63_kernel_read_readvariableop,
(savev2_dense_63_bias_read_readvariableop.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop.
*savev2_dense_64_kernel_read_readvariableop,
(savev2_dense_64_bias_read_readvariableop.
*savev2_dense_62_kernel_read_readvariableop,
(savev2_dense_62_bias_read_readvariableop.
*savev2_dense_60_kernel_read_readvariableop,
(savev2_dense_60_bias_read_readvariableop.
*savev2_dense_65_kernel_read_readvariableop,
(savev2_dense_65_bias_read_readvariableop.
*savev2_dense_66_kernel_read_readvariableop,
(savev2_dense_66_bias_read_readvariableop.
*savev2_dense_67_kernel_read_readvariableop,
(savev2_dense_67_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_44_kernel_m_read_readvariableop4
0savev2_adam_conv2d_44_bias_m_read_readvariableop6
2savev2_adam_conv2d_45_kernel_m_read_readvariableop4
0savev2_adam_conv2d_45_bias_m_read_readvariableop6
2savev2_adam_conv2d_46_kernel_m_read_readvariableop4
0savev2_adam_conv2d_46_bias_m_read_readvariableop6
2savev2_adam_conv2d_47_kernel_m_read_readvariableop4
0savev2_adam_conv2d_47_bias_m_read_readvariableop5
1savev2_adam_dense_63_kernel_m_read_readvariableop3
/savev2_adam_dense_63_bias_m_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableop5
1savev2_adam_dense_64_kernel_m_read_readvariableop3
/savev2_adam_dense_64_bias_m_read_readvariableop5
1savev2_adam_dense_62_kernel_m_read_readvariableop3
/savev2_adam_dense_62_bias_m_read_readvariableop5
1savev2_adam_dense_60_kernel_m_read_readvariableop3
/savev2_adam_dense_60_bias_m_read_readvariableop5
1savev2_adam_dense_65_kernel_m_read_readvariableop3
/savev2_adam_dense_65_bias_m_read_readvariableop5
1savev2_adam_dense_66_kernel_m_read_readvariableop3
/savev2_adam_dense_66_bias_m_read_readvariableop5
1savev2_adam_dense_67_kernel_m_read_readvariableop3
/savev2_adam_dense_67_bias_m_read_readvariableop6
2savev2_adam_conv2d_44_kernel_v_read_readvariableop4
0savev2_adam_conv2d_44_bias_v_read_readvariableop6
2savev2_adam_conv2d_45_kernel_v_read_readvariableop4
0savev2_adam_conv2d_45_bias_v_read_readvariableop6
2savev2_adam_conv2d_46_kernel_v_read_readvariableop4
0savev2_adam_conv2d_46_bias_v_read_readvariableop6
2savev2_adam_conv2d_47_kernel_v_read_readvariableop4
0savev2_adam_conv2d_47_bias_v_read_readvariableop5
1savev2_adam_dense_63_kernel_v_read_readvariableop3
/savev2_adam_dense_63_bias_v_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableop5
1savev2_adam_dense_64_kernel_v_read_readvariableop3
/savev2_adam_dense_64_bias_v_read_readvariableop5
1savev2_adam_dense_62_kernel_v_read_readvariableop3
/savev2_adam_dense_62_bias_v_read_readvariableop5
1savev2_adam_dense_60_kernel_v_read_readvariableop3
/savev2_adam_dense_60_bias_v_read_readvariableop5
1savev2_adam_dense_65_kernel_v_read_readvariableop3
/savev2_adam_dense_65_bias_v_read_readvariableop5
1savev2_adam_dense_66_kernel_v_read_readvariableop3
/savev2_adam_dense_66_bias_v_read_readvariableop5
1savev2_adam_dense_67_kernel_v_read_readvariableop3
/savev2_adam_dense_67_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?.
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?-
value?-B?-RB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?
value?B?RB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_44_kernel_read_readvariableop)savev2_conv2d_44_bias_read_readvariableop+savev2_conv2d_45_kernel_read_readvariableop)savev2_conv2d_45_bias_read_readvariableop+savev2_conv2d_46_kernel_read_readvariableop)savev2_conv2d_46_bias_read_readvariableop+savev2_conv2d_47_kernel_read_readvariableop)savev2_conv2d_47_bias_read_readvariableop*savev2_dense_63_kernel_read_readvariableop(savev2_dense_63_bias_read_readvariableop*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop*savev2_dense_64_kernel_read_readvariableop(savev2_dense_64_bias_read_readvariableop*savev2_dense_62_kernel_read_readvariableop(savev2_dense_62_bias_read_readvariableop*savev2_dense_60_kernel_read_readvariableop(savev2_dense_60_bias_read_readvariableop*savev2_dense_65_kernel_read_readvariableop(savev2_dense_65_bias_read_readvariableop*savev2_dense_66_kernel_read_readvariableop(savev2_dense_66_bias_read_readvariableop*savev2_dense_67_kernel_read_readvariableop(savev2_dense_67_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_44_kernel_m_read_readvariableop0savev2_adam_conv2d_44_bias_m_read_readvariableop2savev2_adam_conv2d_45_kernel_m_read_readvariableop0savev2_adam_conv2d_45_bias_m_read_readvariableop2savev2_adam_conv2d_46_kernel_m_read_readvariableop0savev2_adam_conv2d_46_bias_m_read_readvariableop2savev2_adam_conv2d_47_kernel_m_read_readvariableop0savev2_adam_conv2d_47_bias_m_read_readvariableop1savev2_adam_dense_63_kernel_m_read_readvariableop/savev2_adam_dense_63_bias_m_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop1savev2_adam_dense_64_kernel_m_read_readvariableop/savev2_adam_dense_64_bias_m_read_readvariableop1savev2_adam_dense_62_kernel_m_read_readvariableop/savev2_adam_dense_62_bias_m_read_readvariableop1savev2_adam_dense_60_kernel_m_read_readvariableop/savev2_adam_dense_60_bias_m_read_readvariableop1savev2_adam_dense_65_kernel_m_read_readvariableop/savev2_adam_dense_65_bias_m_read_readvariableop1savev2_adam_dense_66_kernel_m_read_readvariableop/savev2_adam_dense_66_bias_m_read_readvariableop1savev2_adam_dense_67_kernel_m_read_readvariableop/savev2_adam_dense_67_bias_m_read_readvariableop2savev2_adam_conv2d_44_kernel_v_read_readvariableop0savev2_adam_conv2d_44_bias_v_read_readvariableop2savev2_adam_conv2d_45_kernel_v_read_readvariableop0savev2_adam_conv2d_45_bias_v_read_readvariableop2savev2_adam_conv2d_46_kernel_v_read_readvariableop0savev2_adam_conv2d_46_bias_v_read_readvariableop2savev2_adam_conv2d_47_kernel_v_read_readvariableop0savev2_adam_conv2d_47_bias_v_read_readvariableop1savev2_adam_dense_63_kernel_v_read_readvariableop/savev2_adam_dense_63_bias_v_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableop1savev2_adam_dense_64_kernel_v_read_readvariableop/savev2_adam_dense_64_bias_v_read_readvariableop1savev2_adam_dense_62_kernel_v_read_readvariableop/savev2_adam_dense_62_bias_v_read_readvariableop1savev2_adam_dense_60_kernel_v_read_readvariableop/savev2_adam_dense_60_bias_v_read_readvariableop1savev2_adam_dense_65_kernel_v_read_readvariableop/savev2_adam_dense_65_bias_v_read_readvariableop1savev2_adam_dense_66_kernel_v_read_readvariableop/savev2_adam_dense_66_bias_v_read_readvariableop1savev2_adam_dense_67_kernel_v_read_readvariableop/savev2_adam_dense_67_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *`
dtypesV
T2R	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@?:?:??:?:??:?:H:H:D:D:
?(?:?:
?$?:?:
??:?:
??:?:
??:?:	?:: : : : : : : : : :@:@:@?:?:??:?:??:?:H:H:D:D:
?(?:?:
?$?:?:
??:?:
??:?:
??:?:	?::@:@:@?:?:??:?:??:?:H:H:D:D:
?(?:?:
?$?:?:
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:$	 

_output_shapes

:H: 


_output_shapes
:H:$ 

_output_shapes

:D: 

_output_shapes
:D:&"
 
_output_shapes
:
?(?:!

_output_shapes	
:?:&"
 
_output_shapes
:
?$?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
:@: #

_output_shapes
:@:-$)
'
_output_shapes
:@?:!%

_output_shapes	
:?:.&*
(
_output_shapes
:??:!'

_output_shapes	
:?:.(*
(
_output_shapes
:??:!)

_output_shapes	
:?:$* 

_output_shapes

:H: +

_output_shapes
:H:$, 

_output_shapes

:D: -

_output_shapes
:D:&."
 
_output_shapes
:
?(?:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
?$?:!1

_output_shapes	
:?:&2"
 
_output_shapes
:
??:!3

_output_shapes	
:?:&4"
 
_output_shapes
:
??:!5

_output_shapes	
:?:&6"
 
_output_shapes
:
??:!7

_output_shapes	
:?:%8!

_output_shapes
:	?: 9

_output_shapes
::,:(
&
_output_shapes
:@: ;

_output_shapes
:@:-<)
'
_output_shapes
:@?:!=

_output_shapes	
:?:.>*
(
_output_shapes
:??:!?

_output_shapes	
:?:.@*
(
_output_shapes
:??:!A

_output_shapes	
:?:$B 

_output_shapes

:H: C

_output_shapes
:H:$D 

_output_shapes

:D: E

_output_shapes
:D:&F"
 
_output_shapes
:
?(?:!G

_output_shapes	
:?:&H"
 
_output_shapes
:
?$?:!I

_output_shapes	
:?:&J"
 
_output_shapes
:
??:!K

_output_shapes	
:?:&L"
 
_output_shapes
:
??:!M

_output_shapes	
:?:&N"
 
_output_shapes
:
??:!O

_output_shapes	
:?:%P!

_output_shapes
:	?: Q

_output_shapes
::R

_output_shapes
: 
?

?
C__inference_dense_60_layer_call_and_return_conditional_losses_88039

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_31_layer_call_fn_89969

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_31_layer_call_and_return_conditional_losses_88116a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_29_layer_call_and_return_conditional_losses_89797

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????DD:S O
+
_output_shapes
:?????????DD
 
_user_specified_nameinputs
?
?
(__inference_dense_61_layer_call_fn_89717

inputs
unknown:D
	unknown_0:D
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????DD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_61_layer_call_and_return_conditional_losses_87961s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????DD`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????D
 
_user_specified_nameinputs
?
c
E__inference_dropout_57_layer_call_and_return_conditional_losses_88098

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_89534

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????

?*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????

?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_56_layer_call_fn_89758

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_56_layer_call_and_return_conditional_losses_88422x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_61_layer_call_and_return_conditional_losses_87961

inputs3
!tensordot_readvariableop_resource:D-
biasadd_readvariableop_resource:D
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:D*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????D?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????D[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:DY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????DDr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:D*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????DDT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????DDe
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????DDz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????D
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_89668

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_concatenate_7_layer_call_and_return_conditional_losses_88108

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_59_layer_call_and_return_conditional_losses_89895

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_45_layer_call_fn_89503

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_87851x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_57_layer_call_and_return_conditional_losses_89937

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_65_layer_call_and_return_conditional_losses_89995

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_64_layer_call_and_return_conditional_losses_88073

inputs2
matmul_readvariableop_resource:
?(?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?(?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????(
 
_user_specified_nameinputs
?s
?
B__inference_model_7_layer_call_and_return_conditional_losses_88671

inputs
inputs_1
inputs_2)
conv2d_44_88595:@
conv2d_44_88597:@*
conv2d_45_88601:@?
conv2d_45_88603:	?+
conv2d_46_88608:??
conv2d_46_88610:	?+
conv2d_47_88615:??
conv2d_47_88617:	? 
dense_61_88622:D
dense_61_88624:D 
dense_63_88627:H
dense_63_88629:H"
dense_60_88635:
??
dense_60_88637:	?"
dense_62_88640:
?$?
dense_62_88642:	?"
dense_64_88645:
?(?
dense_64_88647:	?"
dense_65_88655:
??
dense_65_88657:	?"
dense_66_88660:
??
dense_66_88662:	?!
dense_67_88665:	?
dense_67_88667:
identity??!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall?"dropout_54/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?"dropout_56/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?"dropout_58/StatefulPartitionedCall?"dropout_59/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_44_88595conv2d_44_88597*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,,@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_87828?
 max_pooling2d_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87838?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_44/PartitionedCall:output:0conv2d_45_88601conv2d_45_88603*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_87851?
 max_pooling2d_45/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87861?
"dropout_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_54_layer_call_and_return_conditional_losses_88498?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall+dropout_54/StatefulPartitionedCall:output:0conv2d_46_88608conv2d_46_88610*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_87881?
 max_pooling2d_46/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87891?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0#^dropout_54/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_55_layer_call_and_return_conditional_losses_88460?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall+dropout_55/StatefulPartitionedCall:output:0conv2d_47_88615conv2d_47_88617*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_87911?
 max_pooling2d_47/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87921?
"dropout_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0#^dropout_55/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_56_layer_call_and_return_conditional_losses_88422?
 dense_61/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_61_88622dense_61_88624*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????DD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_61_layer_call_and_return_conditional_losses_87961?
 dense_63/StatefulPartitionedCallStatefulPartitionedCallinputs_2dense_63_88627dense_63_88629*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????HH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_87998?
flatten_28/PartitionedCallPartitionedCall+dropout_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_88010?
flatten_29/PartitionedCallPartitionedCall)dense_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_29_layer_call_and_return_conditional_losses_88018?
flatten_30/PartitionedCallPartitionedCall)dense_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_30_layer_call_and_return_conditional_losses_88026?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_60_88635dense_60_88637*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_60_layer_call_and_return_conditional_losses_88039?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_62_88640dense_62_88642*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_62_layer_call_and_return_conditional_losses_88056?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_64_88645dense_64_88647*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_88073?
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0#^dropout_56/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_59_layer_call_and_return_conditional_losses_88331?
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall)dense_62/StatefulPartitionedCall:output:0#^dropout_59/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_88308?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0#^dropout_58/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_88285?
concatenate_7/PartitionedCallPartitionedCall+dropout_59/StatefulPartitionedCall:output:0+dropout_58/StatefulPartitionedCall:output:0+dropout_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_88108?
flatten_31/PartitionedCallPartitionedCall&concatenate_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_31_layer_call_and_return_conditional_losses_88116?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall#flatten_31/PartitionedCall:output:0dense_65_88655dense_65_88657*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_88129?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_88660dense_66_88662*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_88146?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_88665dense_67_88667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_88163x
IdentityIdentity)dense_67/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall#^dropout_54/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall#^dropout_56/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2H
"dropout_54/StatefulPartitionedCall"dropout_54/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall2H
"dropout_56/StatefulPartitionedCall"dropout_56/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????D
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
a
E__inference_flatten_29_layer_call_and_return_conditional_losses_88018

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????DD:S O
+
_output_shapes
:?????????DD
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_44_layer_call_fn_89484

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87838h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,,@:W S
/
_output_shapes
:?????????,,@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_89494

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????,,@:W S
/
_output_shapes
:?????????,,@
 
_user_specified_nameinputs
?
c
*__inference_dropout_55_layer_call_fn_89611

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_55_layer_call_and_return_conditional_losses_88460x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?2
!__inference__traced_restore_90556
file_prefix;
!assignvariableop_conv2d_44_kernel:@/
!assignvariableop_1_conv2d_44_bias:@>
#assignvariableop_2_conv2d_45_kernel:@?0
!assignvariableop_3_conv2d_45_bias:	??
#assignvariableop_4_conv2d_46_kernel:??0
!assignvariableop_5_conv2d_46_bias:	??
#assignvariableop_6_conv2d_47_kernel:??0
!assignvariableop_7_conv2d_47_bias:	?4
"assignvariableop_8_dense_63_kernel:H.
 assignvariableop_9_dense_63_bias:H5
#assignvariableop_10_dense_61_kernel:D/
!assignvariableop_11_dense_61_bias:D7
#assignvariableop_12_dense_64_kernel:
?(?0
!assignvariableop_13_dense_64_bias:	?7
#assignvariableop_14_dense_62_kernel:
?$?0
!assignvariableop_15_dense_62_bias:	?7
#assignvariableop_16_dense_60_kernel:
??0
!assignvariableop_17_dense_60_bias:	?7
#assignvariableop_18_dense_65_kernel:
??0
!assignvariableop_19_dense_65_bias:	?7
#assignvariableop_20_dense_66_kernel:
??0
!assignvariableop_21_dense_66_bias:	?6
#assignvariableop_22_dense_67_kernel:	?/
!assignvariableop_23_dense_67_bias:'
assignvariableop_24_adam_iter:	 )
assignvariableop_25_adam_beta_1: )
assignvariableop_26_adam_beta_2: (
assignvariableop_27_adam_decay: 0
&assignvariableop_28_adam_learning_rate: #
assignvariableop_29_total: #
assignvariableop_30_count: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: E
+assignvariableop_33_adam_conv2d_44_kernel_m:@7
)assignvariableop_34_adam_conv2d_44_bias_m:@F
+assignvariableop_35_adam_conv2d_45_kernel_m:@?8
)assignvariableop_36_adam_conv2d_45_bias_m:	?G
+assignvariableop_37_adam_conv2d_46_kernel_m:??8
)assignvariableop_38_adam_conv2d_46_bias_m:	?G
+assignvariableop_39_adam_conv2d_47_kernel_m:??8
)assignvariableop_40_adam_conv2d_47_bias_m:	?<
*assignvariableop_41_adam_dense_63_kernel_m:H6
(assignvariableop_42_adam_dense_63_bias_m:H<
*assignvariableop_43_adam_dense_61_kernel_m:D6
(assignvariableop_44_adam_dense_61_bias_m:D>
*assignvariableop_45_adam_dense_64_kernel_m:
?(?7
(assignvariableop_46_adam_dense_64_bias_m:	?>
*assignvariableop_47_adam_dense_62_kernel_m:
?$?7
(assignvariableop_48_adam_dense_62_bias_m:	?>
*assignvariableop_49_adam_dense_60_kernel_m:
??7
(assignvariableop_50_adam_dense_60_bias_m:	?>
*assignvariableop_51_adam_dense_65_kernel_m:
??7
(assignvariableop_52_adam_dense_65_bias_m:	?>
*assignvariableop_53_adam_dense_66_kernel_m:
??7
(assignvariableop_54_adam_dense_66_bias_m:	?=
*assignvariableop_55_adam_dense_67_kernel_m:	?6
(assignvariableop_56_adam_dense_67_bias_m:E
+assignvariableop_57_adam_conv2d_44_kernel_v:@7
)assignvariableop_58_adam_conv2d_44_bias_v:@F
+assignvariableop_59_adam_conv2d_45_kernel_v:@?8
)assignvariableop_60_adam_conv2d_45_bias_v:	?G
+assignvariableop_61_adam_conv2d_46_kernel_v:??8
)assignvariableop_62_adam_conv2d_46_bias_v:	?G
+assignvariableop_63_adam_conv2d_47_kernel_v:??8
)assignvariableop_64_adam_conv2d_47_bias_v:	?<
*assignvariableop_65_adam_dense_63_kernel_v:H6
(assignvariableop_66_adam_dense_63_bias_v:H<
*assignvariableop_67_adam_dense_61_kernel_v:D6
(assignvariableop_68_adam_dense_61_bias_v:D>
*assignvariableop_69_adam_dense_64_kernel_v:
?(?7
(assignvariableop_70_adam_dense_64_bias_v:	?>
*assignvariableop_71_adam_dense_62_kernel_v:
?$?7
(assignvariableop_72_adam_dense_62_bias_v:	?>
*assignvariableop_73_adam_dense_60_kernel_v:
??7
(assignvariableop_74_adam_dense_60_bias_v:	?>
*assignvariableop_75_adam_dense_65_kernel_v:
??7
(assignvariableop_76_adam_dense_65_bias_v:	?>
*assignvariableop_77_adam_dense_66_kernel_v:
??7
(assignvariableop_78_adam_dense_66_bias_v:	?=
*assignvariableop_79_adam_dense_67_kernel_v:	?6
(assignvariableop_80_adam_dense_67_bias_v:
identity_82??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_9?.
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?-
value?-B?-RB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?
value?B?RB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*`
dtypesV
T2R	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_44_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_44_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_45_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_45_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_46_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_46_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_47_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_47_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_63_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_63_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_61_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_61_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_64_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_64_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_62_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_62_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_60_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_60_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_65_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_65_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_66_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_66_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_67_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_67_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_44_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_44_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_45_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_45_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_46_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_46_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_47_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_47_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_63_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_63_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_61_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_61_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_64_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_64_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_62_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_62_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_60_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_60_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_65_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_65_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_66_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_66_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_67_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_67_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_44_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_44_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_45_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_45_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_46_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_46_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_47_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_47_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_63_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_63_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_61_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_61_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_64_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_64_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_62_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_62_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_60_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_60_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_65_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_65_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_66_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_66_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_67_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_67_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_81Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_82IdentityIdentity_81:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_82Identity_82:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
F
*__inference_dropout_54_layer_call_fn_89539

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_54_layer_call_and_return_conditional_losses_87868i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????

?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????

?:X T
0
_output_shapes
:?????????

?
 
_user_specified_nameinputs
?
F
*__inference_dropout_55_layer_call_fn_89606

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_55_layer_call_and_return_conditional_losses_87898i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_60_layer_call_fn_89857

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_60_layer_call_and_return_conditional_losses_88039p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_30_layer_call_and_return_conditional_losses_88026

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????HH:S O
+
_output_shapes
:?????????HH
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_47_layer_call_fn_89658

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87921i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_57_layer_call_fn_89927

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_88098a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_58_layer_call_and_return_conditional_losses_88091

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_30_layer_call_fn_89780

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_30_layer_call_and_return_conditional_losses_88026a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????HH:S O
+
_output_shapes
:?????????HH
 
_user_specified_nameinputs
?
a
E__inference_flatten_28_layer_call_and_return_conditional_losses_89808

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_47_layer_call_fn_89637

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_87911x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_65_layer_call_fn_89984

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_88129p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_59_layer_call_fn_89873

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_59_layer_call_and_return_conditional_losses_88084a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?j
?

B__inference_model_7_layer_call_and_return_conditional_losses_88858
	convinput
landmarksinput
hoginput)
conv2d_44_88782:@
conv2d_44_88784:@*
conv2d_45_88788:@?
conv2d_45_88790:	?+
conv2d_46_88795:??
conv2d_46_88797:	?+
conv2d_47_88802:??
conv2d_47_88804:	? 
dense_61_88809:D
dense_61_88811:D 
dense_63_88814:H
dense_63_88816:H"
dense_60_88822:
??
dense_60_88824:	?"
dense_62_88827:
?$?
dense_62_88829:	?"
dense_64_88832:
?(?
dense_64_88834:	?"
dense_65_88842:
??
dense_65_88844:	?"
dense_66_88847:
??
dense_66_88849:	?!
dense_67_88852:	?
dense_67_88854:
identity??!conv2d_44/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall? dense_66/StatefulPartitionedCall? dense_67/StatefulPartitionedCall?
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall	convinputconv2d_44_88782conv2d_44_88784*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????,,@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_44_layer_call_and_return_conditional_losses_87828?
 max_pooling2d_44/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87838?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_44/PartitionedCall:output:0conv2d_45_88788conv2d_45_88790*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_87851?
 max_pooling2d_45/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87861?
dropout_54/PartitionedCallPartitionedCall)max_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????

?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_54_layer_call_and_return_conditional_losses_87868?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall#dropout_54/PartitionedCall:output:0conv2d_46_88795conv2d_46_88797*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_87881?
 max_pooling2d_46/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_87891?
dropout_55/PartitionedCallPartitionedCall)max_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_55_layer_call_and_return_conditional_losses_87898?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall#dropout_55/PartitionedCall:output:0conv2d_47_88802conv2d_47_88804*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_87911?
 max_pooling2d_47/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_87921?
dropout_56/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_56_layer_call_and_return_conditional_losses_87928?
 dense_61/StatefulPartitionedCallStatefulPartitionedCalllandmarksinputdense_61_88809dense_61_88811*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????DD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_61_layer_call_and_return_conditional_losses_87961?
 dense_63/StatefulPartitionedCallStatefulPartitionedCallhoginputdense_63_88814dense_63_88816*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????HH*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_87998?
flatten_28/PartitionedCallPartitionedCall#dropout_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_88010?
flatten_29/PartitionedCallPartitionedCall)dense_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_29_layer_call_and_return_conditional_losses_88018?
flatten_30/PartitionedCallPartitionedCall)dense_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_30_layer_call_and_return_conditional_losses_88026?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_60_88822dense_60_88824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_60_layer_call_and_return_conditional_losses_88039?
 dense_62/StatefulPartitionedCallStatefulPartitionedCall#flatten_29/PartitionedCall:output:0dense_62_88827dense_62_88829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_62_layer_call_and_return_conditional_losses_88056?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_64_88832dense_64_88834*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_88073?
dropout_59/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_59_layer_call_and_return_conditional_losses_88084?
dropout_58/PartitionedCallPartitionedCall)dense_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_88091?
dropout_57/PartitionedCallPartitionedCall)dense_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_88098?
concatenate_7/PartitionedCallPartitionedCall#dropout_59/PartitionedCall:output:0#dropout_58/PartitionedCall:output:0#dropout_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_88108?
flatten_31/PartitionedCallPartitionedCall&concatenate_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_31_layer_call_and_return_conditional_losses_88116?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall#flatten_31/PartitionedCall:output:0dense_65_88842dense_65_88844*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_88129?
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_88847dense_66_88849*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_66_layer_call_and_return_conditional_losses_88146?
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_88852dense_67_88854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_67_layer_call_and_return_conditional_losses_88163x
IdentityIdentity)dense_67/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes{
y:?????????00:?????????D:?????????H: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall:Z V
/
_output_shapes
:?????????00
#
_user_specified_name	ConvInput:[W
+
_output_shapes
:?????????D
(
_user_specified_nameLandmarksInput:UQ
+
_output_shapes
:?????????H
"
_user_specified_name
HogInput
?	
d
E__inference_dropout_59_layer_call_and_return_conditional_losses_88331

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_67_layer_call_and_return_conditional_losses_90035

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_66_layer_call_and_return_conditional_losses_90015

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_55_layer_call_and_return_conditional_losses_87898

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_67_layer_call_and_return_conditional_losses_88163

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_87861

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????

?*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????

?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_56_layer_call_and_return_conditional_losses_88422

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_44_layer_call_fn_89479

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_87767?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_56_layer_call_and_return_conditional_losses_89775

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_57_layer_call_and_return_conditional_losses_89949

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_89474

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????,,@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????,,@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????,,@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
	ConvInput:
serving_default_ConvInput:0?????????00
A
HogInput5
serving_default_HogInput:0?????????H
M
LandmarksInput;
 serving_default_LandmarksInput:0?????????D<
dense_670
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer_with_weights-8
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer_with_weights-9
layer-27
layer_with_weights-10
layer-28
layer_with_weights-11
layer-29
	optimizer
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

}kernel
~bias
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate%m?&m?/m?0m?=m?>m?Km?Lm?Um?Vm?[m?\m?qm?rm?wm?xm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?%v?&v?/v?0v?=v?>v?Kv?Lv?Uv?Vv?[v?\v?qv?rv?wv?xv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
%0
&1
/2
03
=4
>5
K6
L7
U8
V9
[10
\11
q12
r13
w14
x15
}16
~17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
?
%0
&1
/2
03
=4
>5
K6
L7
U8
V9
[10
\11
q12
r13
w14
x15
}16
~17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(@2conv2d_44/kernel
:@2conv2d_44/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@?2conv2d_45/kernel
:?2conv2d_45/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_46/kernel
:?2conv2d_46/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_47/kernel
:?2conv2d_47/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:H2dense_63/kernel
:H2dense_63/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:D2dense_61/kernel
:D2dense_61/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?(?2dense_64/kernel
:?2dense_64/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?$?2dense_62/kernel
:?2dense_62/bias
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_60/kernel
:?2dense_60/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_65/kernel
:?2dense_65/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_66/kernel
:?2dense_66/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_67/kernel
:2dense_67/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-@2Adam/conv2d_44/kernel/m
!:@2Adam/conv2d_44/bias/m
0:.@?2Adam/conv2d_45/kernel/m
": ?2Adam/conv2d_45/bias/m
1:/??2Adam/conv2d_46/kernel/m
": ?2Adam/conv2d_46/bias/m
1:/??2Adam/conv2d_47/kernel/m
": ?2Adam/conv2d_47/bias/m
&:$H2Adam/dense_63/kernel/m
 :H2Adam/dense_63/bias/m
&:$D2Adam/dense_61/kernel/m
 :D2Adam/dense_61/bias/m
(:&
?(?2Adam/dense_64/kernel/m
!:?2Adam/dense_64/bias/m
(:&
?$?2Adam/dense_62/kernel/m
!:?2Adam/dense_62/bias/m
(:&
??2Adam/dense_60/kernel/m
!:?2Adam/dense_60/bias/m
(:&
??2Adam/dense_65/kernel/m
!:?2Adam/dense_65/bias/m
(:&
??2Adam/dense_66/kernel/m
!:?2Adam/dense_66/bias/m
':%	?2Adam/dense_67/kernel/m
 :2Adam/dense_67/bias/m
/:-@2Adam/conv2d_44/kernel/v
!:@2Adam/conv2d_44/bias/v
0:.@?2Adam/conv2d_45/kernel/v
": ?2Adam/conv2d_45/bias/v
1:/??2Adam/conv2d_46/kernel/v
": ?2Adam/conv2d_46/bias/v
1:/??2Adam/conv2d_47/kernel/v
": ?2Adam/conv2d_47/bias/v
&:$H2Adam/dense_63/kernel/v
 :H2Adam/dense_63/bias/v
&:$D2Adam/dense_61/kernel/v
 :D2Adam/dense_61/bias/v
(:&
?(?2Adam/dense_64/kernel/v
!:?2Adam/dense_64/bias/v
(:&
?$?2Adam/dense_62/kernel/v
!:?2Adam/dense_62/bias/v
(:&
??2Adam/dense_60/kernel/v
!:?2Adam/dense_60/bias/v
(:&
??2Adam/dense_65/kernel/v
!:?2Adam/dense_65/bias/v
(:&
??2Adam/dense_66/kernel/v
!:?2Adam/dense_66/bias/v
':%	?2Adam/dense_67/kernel/v
 :2Adam/dense_67/bias/v
?2?
'__inference_model_7_layer_call_fn_88221
'__inference_model_7_layer_call_fn_89057
'__inference_model_7_layer_call_fn_89112
'__inference_model_7_layer_call_fn_88777?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_7_layer_call_and_return_conditional_losses_89262
B__inference_model_7_layer_call_and_return_conditional_losses_89454
B__inference_model_7_layer_call_and_return_conditional_losses_88858
B__inference_model_7_layer_call_and_return_conditional_losses_88939?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_87758	ConvInputLandmarksInputHogInput"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_44_layer_call_fn_89463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_44_layer_call_and_return_conditional_losses_89474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_44_layer_call_fn_89479
0__inference_max_pooling2d_44_layer_call_fn_89484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_89489
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_89494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_45_layer_call_fn_89503?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_89514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_45_layer_call_fn_89519
0__inference_max_pooling2d_45_layer_call_fn_89524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_89529
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_89534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_54_layer_call_fn_89539
*__inference_dropout_54_layer_call_fn_89544?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_54_layer_call_and_return_conditional_losses_89549
E__inference_dropout_54_layer_call_and_return_conditional_losses_89561?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_46_layer_call_fn_89570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_46_layer_call_and_return_conditional_losses_89581?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_46_layer_call_fn_89586
0__inference_max_pooling2d_46_layer_call_fn_89591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_89596
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_89601?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_55_layer_call_fn_89606
*__inference_dropout_55_layer_call_fn_89611?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_55_layer_call_and_return_conditional_losses_89616
E__inference_dropout_55_layer_call_and_return_conditional_losses_89628?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_47_layer_call_fn_89637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_47_layer_call_and_return_conditional_losses_89648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_47_layer_call_fn_89653
0__inference_max_pooling2d_47_layer_call_fn_89658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_89663
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_89668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_63_layer_call_fn_89677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_63_layer_call_and_return_conditional_losses_89708?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_61_layer_call_fn_89717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_61_layer_call_and_return_conditional_losses_89748?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_56_layer_call_fn_89753
*__inference_dropout_56_layer_call_fn_89758?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_56_layer_call_and_return_conditional_losses_89763
E__inference_dropout_56_layer_call_and_return_conditional_losses_89775?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_flatten_30_layer_call_fn_89780?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_30_layer_call_and_return_conditional_losses_89786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_29_layer_call_fn_89791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_29_layer_call_and_return_conditional_losses_89797?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_28_layer_call_fn_89802?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_28_layer_call_and_return_conditional_losses_89808?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_64_layer_call_fn_89817?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_64_layer_call_and_return_conditional_losses_89828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_62_layer_call_fn_89837?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_62_layer_call_and_return_conditional_losses_89848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_60_layer_call_fn_89857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_60_layer_call_and_return_conditional_losses_89868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_59_layer_call_fn_89873
*__inference_dropout_59_layer_call_fn_89878?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_59_layer_call_and_return_conditional_losses_89883
E__inference_dropout_59_layer_call_and_return_conditional_losses_89895?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_58_layer_call_fn_89900
*__inference_dropout_58_layer_call_fn_89905?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_58_layer_call_and_return_conditional_losses_89910
E__inference_dropout_58_layer_call_and_return_conditional_losses_89922?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_57_layer_call_fn_89927
*__inference_dropout_57_layer_call_fn_89932?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_57_layer_call_and_return_conditional_losses_89937
E__inference_dropout_57_layer_call_and_return_conditional_losses_89949?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_concatenate_7_layer_call_fn_89956?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_7_layer_call_and_return_conditional_losses_89964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_31_layer_call_fn_89969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_31_layer_call_and_return_conditional_losses_89975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_65_layer_call_fn_89984?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_65_layer_call_and_return_conditional_losses_89995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_66_layer_call_fn_90004?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_66_layer_call_and_return_conditional_losses_90015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_67_layer_call_fn_90024?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_67_layer_call_and_return_conditional_losses_90035?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_89002	ConvInputHogInputLandmarksInput"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_87758?%&/0=>KL[\UV}~wxqr?????????
???
???
+?(
	ConvInput?????????00
,?)
LandmarksInput?????????D
&?#
HogInput?????????H
? "3?0
.
dense_67"?
dense_67??????????
H__inference_concatenate_7_layer_call_and_return_conditional_losses_89964???~
w?t
r?o
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
? "&?#
?
0??????????
? ?
-__inference_concatenate_7_layer_call_fn_89956???~
w?t
r?o
#? 
inputs/0??????????
#? 
inputs/1??????????
#? 
inputs/2??????????
? "????????????
D__inference_conv2d_44_layer_call_and_return_conditional_losses_89474l%&7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????,,@
? ?
)__inference_conv2d_44_layer_call_fn_89463_%&7?4
-?*
(?%
inputs?????????00
? " ??????????,,@?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_89514m/07?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_45_layer_call_fn_89503`/07?4
-?*
(?%
inputs?????????@
? "!????????????
D__inference_conv2d_46_layer_call_and_return_conditional_losses_89581n=>8?5
.?+
)?&
inputs?????????

?
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_46_layer_call_fn_89570a=>8?5
.?+
)?&
inputs?????????

?
? "!????????????
D__inference_conv2d_47_layer_call_and_return_conditional_losses_89648nKL8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_47_layer_call_fn_89637aKL8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_dense_60_layer_call_and_return_conditional_losses_89868^}~0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_60_layer_call_fn_89857Q}~0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_61_layer_call_and_return_conditional_losses_89748d[\3?0
)?&
$?!
inputs?????????D
? ")?&
?
0?????????DD
? ?
(__inference_dense_61_layer_call_fn_89717W[\3?0
)?&
$?!
inputs?????????D
? "??????????DD?
C__inference_dense_62_layer_call_and_return_conditional_losses_89848^wx0?-
&?#
!?
inputs??????????$
? "&?#
?
0??????????
? }
(__inference_dense_62_layer_call_fn_89837Qwx0?-
&?#
!?
inputs??????????$
? "????????????
C__inference_dense_63_layer_call_and_return_conditional_losses_89708dUV3?0
)?&
$?!
inputs?????????H
? ")?&
?
0?????????HH
? ?
(__inference_dense_63_layer_call_fn_89677WUV3?0
)?&
$?!
inputs?????????H
? "??????????HH?
C__inference_dense_64_layer_call_and_return_conditional_losses_89828^qr0?-
&?#
!?
inputs??????????(
? "&?#
?
0??????????
? }
(__inference_dense_64_layer_call_fn_89817Qqr0?-
&?#
!?
inputs??????????(
? "????????????
C__inference_dense_65_layer_call_and_return_conditional_losses_89995`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
(__inference_dense_65_layer_call_fn_89984S??0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_66_layer_call_and_return_conditional_losses_90015`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
(__inference_dense_66_layer_call_fn_90004S??0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_67_layer_call_and_return_conditional_losses_90035_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
(__inference_dense_67_layer_call_fn_90024R??0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dropout_54_layer_call_and_return_conditional_losses_89549n<?9
2?/
)?&
inputs?????????

?
p 
? ".?+
$?!
0?????????

?
? ?
E__inference_dropout_54_layer_call_and_return_conditional_losses_89561n<?9
2?/
)?&
inputs?????????

?
p
? ".?+
$?!
0?????????

?
? ?
*__inference_dropout_54_layer_call_fn_89539a<?9
2?/
)?&
inputs?????????

?
p 
? "!??????????

??
*__inference_dropout_54_layer_call_fn_89544a<?9
2?/
)?&
inputs?????????

?
p
? "!??????????

??
E__inference_dropout_55_layer_call_and_return_conditional_losses_89616n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
E__inference_dropout_55_layer_call_and_return_conditional_losses_89628n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
*__inference_dropout_55_layer_call_fn_89606a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
*__inference_dropout_55_layer_call_fn_89611a<?9
2?/
)?&
inputs??????????
p
? "!????????????
E__inference_dropout_56_layer_call_and_return_conditional_losses_89763n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
E__inference_dropout_56_layer_call_and_return_conditional_losses_89775n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
*__inference_dropout_56_layer_call_fn_89753a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
*__inference_dropout_56_layer_call_fn_89758a<?9
2?/
)?&
inputs??????????
p
? "!????????????
E__inference_dropout_57_layer_call_and_return_conditional_losses_89937^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_57_layer_call_and_return_conditional_losses_89949^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_57_layer_call_fn_89927Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_57_layer_call_fn_89932Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_dropout_58_layer_call_and_return_conditional_losses_89910^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_58_layer_call_and_return_conditional_losses_89922^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_58_layer_call_fn_89900Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_58_layer_call_fn_89905Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_dropout_59_layer_call_and_return_conditional_losses_89883^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_59_layer_call_and_return_conditional_losses_89895^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_59_layer_call_fn_89873Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_59_layer_call_fn_89878Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_flatten_28_layer_call_and_return_conditional_losses_89808b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
*__inference_flatten_28_layer_call_fn_89802U8?5
.?+
)?&
inputs??????????
? "????????????
E__inference_flatten_29_layer_call_and_return_conditional_losses_89797]3?0
)?&
$?!
inputs?????????DD
? "&?#
?
0??????????$
? ~
*__inference_flatten_29_layer_call_fn_89791P3?0
)?&
$?!
inputs?????????DD
? "???????????$?
E__inference_flatten_30_layer_call_and_return_conditional_losses_89786]3?0
)?&
$?!
inputs?????????HH
? "&?#
?
0??????????(
? ~
*__inference_flatten_30_layer_call_fn_89780P3?0
)?&
$?!
inputs?????????HH
? "???????????(?
E__inference_flatten_31_layer_call_and_return_conditional_losses_89975Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
*__inference_flatten_31_layer_call_fn_89969M0?-
&?#
!?
inputs??????????
? "????????????
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_89489?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_44_layer_call_and_return_conditional_losses_89494h7?4
-?*
(?%
inputs?????????,,@
? "-?*
#? 
0?????????@
? ?
0__inference_max_pooling2d_44_layer_call_fn_89479?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_44_layer_call_fn_89484[7?4
-?*
(?%
inputs?????????,,@
? " ??????????@?
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_89529?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_89534j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????

?
? ?
0__inference_max_pooling2d_45_layer_call_fn_89519?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_45_layer_call_fn_89524]8?5
.?+
)?&
inputs??????????
? "!??????????

??
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_89596?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_89601j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_max_pooling2d_46_layer_call_fn_89586?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_46_layer_call_fn_89591]8?5
.?+
)?&
inputs??????????
? "!????????????
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_89663?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_89668j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
0__inference_max_pooling2d_47_layer_call_fn_89653?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_47_layer_call_fn_89658]8?5
.?+
)?&
inputs??????????
? "!????????????
B__inference_model_7_layer_call_and_return_conditional_losses_88858?%&/0=>KL[\UV}~wxqr?????????
???
???
+?(
	ConvInput?????????00
,?)
LandmarksInput?????????D
&?#
HogInput?????????H
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_7_layer_call_and_return_conditional_losses_88939?%&/0=>KL[\UV}~wxqr?????????
???
???
+?(
	ConvInput?????????00
,?)
LandmarksInput?????????D
&?#
HogInput?????????H
p

 
? "%?"
?
0?????????
? ?
B__inference_model_7_layer_call_and_return_conditional_losses_89262?%&/0=>KL[\UV}~wxqr?????????
???
?|
*?'
inputs/0?????????00
&?#
inputs/1?????????D
&?#
inputs/2?????????H
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_7_layer_call_and_return_conditional_losses_89454?%&/0=>KL[\UV}~wxqr?????????
???
?|
*?'
inputs/0?????????00
&?#
inputs/1?????????D
&?#
inputs/2?????????H
p

 
? "%?"
?
0?????????
? ?
'__inference_model_7_layer_call_fn_88221?%&/0=>KL[\UV}~wxqr?????????
???
???
+?(
	ConvInput?????????00
,?)
LandmarksInput?????????D
&?#
HogInput?????????H
p 

 
? "???????????
'__inference_model_7_layer_call_fn_88777?%&/0=>KL[\UV}~wxqr?????????
???
???
+?(
	ConvInput?????????00
,?)
LandmarksInput?????????D
&?#
HogInput?????????H
p

 
? "???????????
'__inference_model_7_layer_call_fn_89057?%&/0=>KL[\UV}~wxqr?????????
???
?|
*?'
inputs/0?????????00
&?#
inputs/1?????????D
&?#
inputs/2?????????H
p 

 
? "???????????
'__inference_model_7_layer_call_fn_89112?%&/0=>KL[\UV}~wxqr?????????
???
?|
*?'
inputs/0?????????00
&?#
inputs/1?????????D
&?#
inputs/2?????????H
p

 
? "???????????
#__inference_signature_wrapper_89002?%&/0=>KL[\UV}~wxqr?????????
? 
???
8
	ConvInput+?(
	ConvInput?????????00
2
HogInput&?#
HogInput?????????H
>
LandmarksInput,?)
LandmarksInput?????????D"3?0
.
dense_67"?
dense_67?????????