# To allow for transferring functions from python to monetdb without the source code we analyze the underlying PyCodeObject structure of a python function object
# We can then convert this PyCodeObject to an encoded string, and decode it to reconstruct the PyCodeObject in MonetDB

# This is necessary because Python throws away the source code of functions created in the interpreter
#   so if we want to pass an arbitrary function to MonetDB we need to transfer it using its underlying Code Object

START TRANSACTION;

CREATE TABLE rval(i integer,j integer);
INSERT INTO rval VALUES (1,4), (2,3), (3,2), (4,1);


CREATE FUNCTION pyapi14_function_format() returns table(s string) language P
{
	# helper function, concat a list of constants used together with the type of the constant to a structure like (int:20)(int:33)(str:test)
	def consts_to_string(consts):
		result = ""
		for const in consts:
			result = result + '(' + str(type(const).__name__) + ':' + str(const) + ')'
		return result
	# helper function, concat names with a comma between them
	def names_to_string(names):
		result = ""
		for name in names:
			result = result + name + ','
		return result
	# helper function, convert string values that cannot be displayed (like \x00) to the string literal "\\x00" so we can transfer it
	def format_code(code):
		result = "";
		for i in code:
			if ord(i) < 16: result = result + '\\x0' + hex(ord(i))[2:]
			else: result = result + '\\x' + hex(ord(i))[2:]
		return result
	# this function converts the function object
	def function_to_string(fun):
		args = fun.__code__.co_argcount
		nlocals = fun.__code__.co_nlocals
		stacksize = fun.__code__.co_stacksize
		flags = fun.__code__.co_flags
		code = format_code(fun.__code__.co_code)
		consts = fun.__code__.co_consts
		names = fun.__code__.co_names
		varnames = fun.__code__.co_varnames
		freevars = fun.__code__.co_freevars
		cellvars = fun.__code__.co_cellvars
		filename = fun.__code__.co_filename
		name = fun.__code__.co_name
		firstlineno = fun.__code__.co_firstlineno
		lnotab = format_code(fun.__code__.co_lnotab)
		return str(args) + '@' + str(nlocals) + '@' + str(stacksize) + '@' + str(flags) + '@' + code + '@' + consts_to_string(consts) + '@' + names_to_string(names) + '@' + names_to_string(varnames) + '@' + names_to_string(freevars) + '@' + names_to_string(cellvars) + '@' + filename + '@' + name + '@' + str(firstlineno) + '@' + lnotab + '@'

	def myfun(a,b):
		return a * b

	def myfun2(a,b):
		return a * 20 + b

	def myfun3(a,b):
		import math
		return numpy.power(a,b)

	return [function_to_string(myfun), 
			function_to_string(myfun2),
			function_to_string(myfun3)];
};

SELECT * FROM pyapi14_function_format();

# note: when creating the SQL function the format is '{@<function_code>};', the first @ symbolizes to the parser that it is not a regular function but an encoded code object

# def myfun(a,b):
#    return a * b
CREATE FUNCTION pyapi14(a integer, b integer) returns integer language P {@2@2@2@67@|\x00\x00|\x01\x00\x14S@(NoneType:None)@@a,b,@@@<stdin>@myfun@1@\x00\x01@};
SELECT pyapi14(i,j) FROM rval;
DROP FUNCTION pyapi14;

# def myfun2(a,b):
#    return a * 20 + b
CREATE FUNCTION pyapi14(a integer, b integer) returns integer language P {@2@2@2@83@\\x7c\\x00\\x00\\x64\\x01\\x00\\x14\\x7c\\x01\\x00\\x17\\x53@(NoneType:None)(int:20)@@a,b,@@@<string>@myfun2@36@\\x00\\x01@};
SELECT pyapi14(i,j) FROM rval;
DROP FUNCTION pyapi14;

#def myfun3(a,b):
#	import math
#	return numpy.power(a,b)
CREATE FUNCTION pyapi14(a integer, b integer) returns integer language P {@2@3@3@83@\\x64\\x01\\x00\\x64\\x00\\x00\\x6c\\x00\\x00\\x7d\\x02\\x00\\x74\\x01\\x00\\x6a\\x02\\x00\\x7c\\x00\\x00\\x7c\\x01\\x00\\x83\\x02\\x00\\x53@(NoneType:None)(int:-1)@math,numpy,power,@a,b,math,@@@<string>@myfun3@38@\\x00\\x01\\x0c\\x01@};
SELECT pyapi14(i,j) FROM rval;
DROP FUNCTION pyapi14;

DROP TABLE rval;


ROLLBACK;
