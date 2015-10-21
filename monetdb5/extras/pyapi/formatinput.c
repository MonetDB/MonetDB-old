
#include "formatinput.h"
#include "type_conversion.h"

#include "monetdb_config.h"
#include "mal.h"
#include "mal_stack.h"
#include "mal_linker.h"
#include "gdk_utils.h"
#include "gdk.h"
#include "mal_exception.h"

const size_t additional_argcount = 4;
const char * additional_args[] = {"_columns", "_column_types", "_values", "_conn"};

//! Parse a PyCodeObject from a string, the string is expected to be in the format {@<encoded_function>};, where <encoded_function> is all the PyCodeObject properties in order
PyObject *PyCodeObject_ParseString(char *string, char **msg);

char* GetArg(char *string, char *storage, int index);
char* GetArg(char *string, char *storage, int index)
{
    int i, j = 0, k = 0;
    int brackets = 0;
    int length = strlen(string);
    for(i = 2; i < length; i++) {
        if (string[i] == '(') brackets++;
        else if (string[i] == ')') brackets--;
        if (brackets == 0 && string[i] == '@') {
            j++;
            if (j > index) {
                storage[k] = '\0';
                break;
            }
        }
        else if (j == index) {
            storage[k++] = string[i];
        }
    }
    return storage;
}

size_t ModifyCode(char *string, char *storage);
size_t ModifyCode(char *string, char *storage)
{
    int i = 0, j = 0, token = 0;
    int length = strlen(string);
    char tokens[3]; tokens[2] = '\0';
    for(i = 0; i < length; i++) {
        if (string[i] == '\\') {
            token = 1;
        } else if (token == 1) {
            token++;
            if (string[i] != 'x') token = 0;
        } else if (token > 1 && token < 4) {
            tokens[token - 2] = string[i];
            token++;
            if (token == 4) {
                storage[j++] = strtol(tokens, NULL, 16);
            }
        } else {
            storage[j++] = string[i];
        }
    }
    storage[j] = '\0';
    return j;
}

PyObject *GetConstantObject(char *string, char *storage, char **msg);
PyObject *GetConstantObject(char *string, char *storage, char **msg)
{
    int numbers = 0, i = 0, j = 0, k = 0;
    int length = strlen(string);
    PyObject *result;
    bool type = false;
    char type_str[100];
    int brackets = 0;
    //first get the amount of constant objects
    for(i = 0; i < length; i++) {
        if (string[i] == '(') brackets++;
        if (string[i] == ')') {
            brackets--;
            if (brackets == 0) numbers++;
        }
    }
    if (brackets != 0) {
        //invalid number of brackets
        *msg = createException(MAL, "pyapi.eval", "Invalid number of brackets in encoded Python code object string.");
        return NULL;
    }

    //then create the python tuple and fill it with the actual python objects
    result = PyTuple_New(numbers);
    for(i = 0; i < length; i++) {
        if (string[i] == '(') brackets++;
        if (string[i] == ')') {
            brackets--;
            if (brackets == 0) {
                PyObject *object = NULL;
                storage[k] = '\0';
                //parse object type
                if (strcmp(type_str, "NoneType") == 0) {
                    object = Py_None;
                } else if (strcmp(type_str, "int") == 0) {
                    object = PyInt_FromString(storage, NULL, 0);
                } else if (strcmp(type_str, "long") == 0) {
                    object = PyLong_FromString(storage, NULL, 0);
                } else if (strcmp(type_str, "float") == 0) {
                    dbl d;
                    str_to_dbl(storage, strlen(storage), &d);
                    object = PyFloat_FromDouble(d);
                } else if (strcmp(type_str, "str") == 0) {
                    object = PyString_FromString(storage);
                } else if (strcmp(type_str, "unicode") == 0) {
                    object = PyUnicode_FromString(storage);
                } else if (strcmp(type_str, "code") == 0) {
                    //recursive call, we've got a function within this function, so we have to parse another code object
                    object = PyCodeObject_ParseString(storage, msg);
                } else {
                    *msg = createException(MAL, "pyapi.eval", "Unrecognized constant type %s in encoded Python code object string.", type_str);
                    return NULL;
                }
                PyTuple_SetItem(result, j, object);

                type = false;
                j++;
                k = 0;    
                continue;            
            }
        } 

        if (string[i] == ':' && !type) {
            type = true;
            type_str[k] = '\0';
            k = 0;
        } else {
            if (type) {
                storage[k++] = string[i];
            } else {
                if (string[i] != '(') {
                    type_str[k++] = string[i];
                }
            }
        }
    }
    return result;
}

PyObject* GetStringTuple(char *string, char *storage);
PyObject* GetStringTuple(char *string, char *storage)
{
    int numbers = 0, i = 0, j = 0, k = 0;
    int length = strlen(string);
    PyObject *result;
    //first get the amount of entries in the string by looking at the amount of commas
    for(i = 0; i < length; i++) {
        if (string[i] == ',') {
            numbers++;
        }
    }
    //then create the pytuple and actually create every PyStringObject
    result = PyTuple_New(numbers); 
    for(i = 0; i < length; i++) {
        if (string[i] == ',' || i == length - 1) {
            PyObject *obj;
            storage[k] = '\0';
            obj = PyString_FromString(storage);
            PyTuple_SetItem(result ,j, obj);
            j++;
            k = 0;
        } else {
            storage[k++] = string[i];
        }
    }
    return result;
}

PyObject *PyCodeObject_ParseString(char *string, char **msg)
{
    int argcount, nlocals, stacksize, flags, firstlineno;
    PyObject *code, *name, *filename, *lnotab;
    PyObject *consts, *names, *varnames, *freevars, *cellvars;
    size_t size;
    char *temp_string = GDKmalloc(strlen(string));
    char *temp_string2 = GDKmalloc(strlen(string));
    if (temp_string == NULL || temp_string2 == NULL) {
        *msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        return NULL;
    }

    //argcount is a single int
    argcount = atoi(GetArg(string, temp_string, 0));
    //nlocals is a single int
    nlocals = atoi(GetArg(string, temp_string, 1));
    //stacksize is a single int
    stacksize = atoi(GetArg(string, temp_string, 2));
    //flags is a single int
    flags = atoi(GetArg(string, temp_string, 3));
    //now parse the code, the code has tokens like "\\x01", which have to be changed to a single character literal '\x01', this is done in the ModifyCode function
    size = ModifyCode(GetArg(string, temp_string, 4), temp_string2);
    code = PyString_FromStringAndSize(temp_string2, size);
    //now parse the constants, constants are a list of python objects in the form of (type:value) (ex: (int:20)(int:33)(str:hello))
    consts = GetConstantObject(GetArg(string, temp_string, 5), temp_string2, msg);
    //now parse the names, this is a list of strings delimited by commas (ex: name,name2,)
    names = GetStringTuple(GetArg(string, temp_string, 6), temp_string2);
    //now parse the varnames, same as above
    varnames = GetStringTuple(GetArg(string, temp_string, 7), temp_string2);
    //now parse the freevars, same as above
    freevars = GetStringTuple(GetArg(string, temp_string, 8), temp_string2);
    //now parse the cellvars, same as above
    cellvars = GetStringTuple(GetArg(string, temp_string, 9), temp_string2);
    //now parse the filename, a single string
    filename = PyString_FromString(GetArg(string, temp_string, 10));
    //now parse the function name, a single string
    name = PyString_FromString(GetArg(string, temp_string, 11));
    //now parse the line number, a single int
    firstlineno = atoi(GetArg(string, temp_string, 12));
    //now parse lnotab, this is a mapping from <addr> -> <lineno> for error reporting purposes, it also has tokens like "\\x01"
    size = ModifyCode(GetArg(string, temp_string, 13), temp_string2);
    lnotab = PyString_FromStringAndSize(temp_string2, size);

    GDKfree(temp_string); GDKfree(temp_string2);
    return (PyObject*)PyCode_New(argcount, nlocals, stacksize, flags, code, consts, names, varnames, freevars, cellvars, filename, name, firstlineno, lnotab);
}

char* FormatCode(char* code, char **args, size_t argcount, size_t tabwidth, PyObject **code_object, char **msg)
{
    // Format the python code by fixing the indentation levels
    // We do two passes, first we get the length of the resulting formatted code and then we actually create the resulting code
    size_t i = 0, j = 0, k = 0;
    size_t length = strlen(code);
    size_t size = 0;
    size_t spaces_per_level = 2;

    size_t code_location = 0;
    char *newcode = NULL;

    size_t indentation_count = 0;
    size_t max_indentation = 100;
    // This keeps track of the different indentation levels
    // indentation_levels is a sorted array with how many spaces of indentation that specific array has
    // so indentation_levels[0] = 4 means that the first level (level 0) has 4 spaces in the source code
    // after this array is constructed we can count the amount of spaces before a statement and look in this
    // array to immediately find the indentation level of the statement
    size_t *indentation_levels;
    // statements_per_level keeps track of how many statements are at the specified indentation level
    // this is needed to compute the size of the resulting formatted code
    // for every indentation level i, we add statements_per_level[i] * (i + 1) * spaces_per_level spaces
    size_t *statements_per_level; 

    size_t initial_spaces = 0;
    size_t statement_size = 0;
    bool seen_statement = false;
    bool multiline_statement = false;
    int multiline_quotes = 0;

    char base_start[] = "def pyfun(";
    char base_end[] = "):\n";
    *msg = NULL;
    if (code[1] == '@') {
        *code_object = PyCodeObject_ParseString(code, msg);
        return NULL;
    }

    indentation_levels = (size_t*)GDKzalloc(max_indentation * sizeof(size_t));
    statements_per_level = (size_t*)GDKzalloc(max_indentation * sizeof(size_t));
    if (indentation_levels == NULL || statements_per_level == NULL) {
        *msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        goto finally;
    }

    // Base function definition size
    // For every argument, add a comma, and add another entry for the '\0'
    size += strlen(base_start) + strlen(base_end) + argcount + 1;
    for(i = 0; i < argcount; i++) {
        if (args[i] != NULL) {
            size += strlen(args[i]) + 1; 
        }
    }
    // Additional parameters
    for(i = 0; i < additional_argcount; i++) size += strlen(additional_args[i]) + 1;

    // First remove the "{" at the start and the "};" at the end of the function, this is added when we have a function created through SQL and python doesn't like them
    // We need to be careful to only remove ones at the start/end, otherwise we might invalidate some otherwise valid python code containing them
    for(i = length - 1, j = 0; i > 0; i--)
    {
        if (code[i] != '\n' && code[i] != ' ' && code[i] != '\t' && code[i] != ';' && code[i] != '}') break;
        if (j == 0) {
            if (code[i] == ';') {
                code[i] = ' ';
                j = 1;
            }
        }
        else if (j == 1) {
            if (code[i] == '}') {
                code[i] = ' ';
                break;
            }
        }
    }
    for(i = 0; i < length; i++) {
        if (code[i] != '\n' && code[i] != ' ' && code[i] != '\t' && code[i] != '{') break;
        if (code[i] == '{') {
            code[i] = ' ';
        }
    }
    // We indent using spaces, four spaces per level
    // We also erase empty lines
    for(i = 0; i < length; i++) {
        // handle multiline strings (strings that start with """)
        if (code[i] == '\"') {
            if (!multiline_statement) {
                multiline_quotes++;
                multiline_statement = multiline_quotes == 3;
            } else {
                multiline_quotes--;
                multiline_statement = multiline_quotes != 0;
            }
        } else {
            multiline_quotes = multiline_statement ? 3 : 0;
        }

        if (!seen_statement) {
            // We have not seen a statement on this line yet
            if (code[i] == '\n'){ 
                // Empty line, skip to the next one
                initial_spaces = 0;
            } else if (code[i] == ' ') {
                initial_spaces++;
            } else if (code[i] == '\t') {
                initial_spaces += tabwidth;
            } else {
                // Statement starts here
                seen_statement = true;
            }
        }
        if (seen_statement) {
            // We have seen a statement on this line, check the indentation level
            statement_size++;

            if (code[i] == '\n' || i == length - 1) {
                // Statement ends here
                bool placed = false;
                size_t level = 0;

                if (multiline_statement) {
                    //if we are in a multiline statement, we don't want to mess with the indentation
                    size += statement_size;
                    initial_spaces = 0;
                    statement_size = 0;
                    continue;
                }
                // First put the indentation in the indentation table
                if (indentation_count >= max_indentation) {
                    // If there is no room in the indentation arrays we will extend them
                    // This probably will never happen unless in really extreme code (or if max_indentation is set very low)
                    size_t *new_indentation = GDKzalloc(2 * max_indentation * sizeof(size_t));
                    size_t *new_statements_per_level;
                    if (new_indentation == NULL) { 
                        *msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
                        goto finally;
                    }
                    new_statements_per_level = GDKzalloc(2 * max_indentation * sizeof(size_t));
                    if (new_statements_per_level == NULL) {
                        *msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
                        goto finally;
                    }

                    for(i = 0; i < max_indentation; i++) {
                        new_indentation[i] = indentation_levels[i];
                        new_statements_per_level[i] = statements_per_level[i];
                    }
                    GDKfree(indentation_levels);
                    GDKfree(statements_per_level);
                    indentation_levels = new_indentation;
                    statements_per_level = new_statements_per_level;
                    max_indentation *= 2;
                }

                for(j = 0; j < indentation_count; j++) {
                    if (initial_spaces == indentation_levels[j]) {
                        // The exact space count is already in the array, so we can stop
                        level = j;
                        placed = true;
                        break;
                    }

                    if (initial_spaces < indentation_levels[j]) {
                        // The indentation level is smaller than this level (but bigger than the previous level)
                        // So the indentation level belongs here, so we move every level past this one upward one level
                        // and put the indentation level here
                        for(k = indentation_count; k > j; k--) {
                            indentation_levels[k] = indentation_levels[k - 1];
                            statements_per_level[k] = statements_per_level[k - 1];
                        }
                        indentation_count++;
                        statements_per_level[j] = 0;
                        indentation_levels[j] = initial_spaces;
                        level = j;
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    // The space count is the biggest we have seen, so we add it to the end of the array
                    level = indentation_count;
                    indentation_levels[indentation_count++] = initial_spaces;
                }
                statements_per_level[level]++;
                size += statement_size;
                seen_statement = false;
                initial_spaces = 0;
                statement_size = 0;
            }
        }
    }
    // Add the amount of spaces we will add to the size
    for(i = 0; i < indentation_count; i++) {
        size += (i + 1) * spaces_per_level * statements_per_level[i];
    }

    // Allocate space for the function
    newcode = GDKzalloc(size);
    if (newcode == NULL) { 
        *msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        goto finally;
    }
    initial_spaces = 0;
    seen_statement = false;

    // First print in the function definition and arguments
    for(i = 0; i < strlen(base_start); i++) {
        newcode[code_location++] = base_start[i];
    }
    // Add user-defined parameters
    for(i = 0; i < argcount; i++) {
        if (args[i] != NULL) {
            for(j = 0; j < strlen(args[i]); j++) {
                newcode[code_location++] = args[i][j];
            }
            if (i != argcount - 1 || additional_argcount > 0) {
                newcode[code_location++] = ',';
            }
        }
    }
    // Add additional parameters
    for(i = 0; i < additional_argcount; i++) {
    if (additional_args[i] != NULL) {
            for(j = 0; j < strlen(additional_args[i]); j++) {
                newcode[code_location++] = additional_args[i][j];
            }
            if (i != additional_argcount - 1) {
                newcode[code_location++] = ',';
            }
        }
    }
    for(i = 0; i < strlen(base_end); i++) {
        newcode[code_location++] = base_end[i];
    }

    // Now the second pass, actually construct the code
    for(i = 0; i < length; i++) {
        //handle multiline statements
        if (code[i] == '\"') {
            if (!multiline_statement) {
                multiline_quotes++;
                multiline_statement = multiline_quotes == 3;
            } else {
                multiline_quotes--;
                multiline_statement = multiline_quotes != 0;
            }
        } else {
            multiline_quotes = multiline_statement ? 3 : 0;
        }

        if (!seen_statement) {
            if (multiline_statement) seen_statement = true; //if we are in a multiline string, we simply want to copy everything (including indentation)
            // We have not seen a statement on this line yet
            else if (code[i] == '\n'){ 
                // Empty line, skip to the next one
                initial_spaces = 0;
            } else if (code[i] == ' ') {
                initial_spaces++;
            } else if (code[i] == '\t') {
                initial_spaces += tabwidth;
            } else {
                // Statement starts here
                seen_statement = true;
                // Look through the indentation_levels array to find the level of the statement
                // from the amount of initial spaces
                bool placed = false;
                int level = 0;
                for(j = 0; j < indentation_count; j++) {
                    if (initial_spaces == indentation_levels[j]) {
                        level = j;
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    // This should never happen, because it means the initial spaces was not present in the array
                    // When we just did exactly the same loop over the array, we should have encountered this statement
                    // This means that something happened to either the indentation_levels array or something happened to the code
                    *msg = createException(MAL, "pyapi.eval", "If you see this error something went wrong in the code. Sorry.");
                    goto finally;
                }
                for(j = 0; j < (level + 1) * spaces_per_level; j++) {
                    // Add spaces to the code
                    newcode[code_location++] = ' ';
                }
            }
        }
        if (seen_statement) {
            // We have seen a statement on this line, copy it
            newcode[code_location++] = code[i];
            if (code[i] == '\n') {
                // The statement has ended, move on to the next line
                seen_statement = false;
                initial_spaces = 0;
                statement_size = 0;
            }
        }
    }
    newcode[code_location] = '\0';
    if (code_location >= size) {
        // Something went wrong with our size computation, this also should never happen
        *msg = createException(MAL, "pyapi.eval", "If you see this error something went wrong in the code (size computation). Sorry.");
        goto finally;
    }
finally:
    GDKfree(indentation_levels);
    GDKfree(statements_per_level);
    return newcode;
}
