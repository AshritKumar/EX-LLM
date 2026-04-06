import sympy as sym

# Initialize pretty printing for better display
sym.init_printing(use_unicode=True)

################### product rule ##############################

# create symbolic variables in sympy
x = sym.symbols('x')

fx = 2*x**2
gx = 4*x**3 - 3*x**4

df = sym.diff(fx, x)  # specify variable for clarity
dg = sym.diff(gx, x)

print("Functions")
# Use SymPy's pprint for pretty printing
sym.pprint(fx)
sym.pprint(gx)

print("\nDerivatives")
sym.pprint(df)
sym.pprint(dg)

# # Alternative: Print as LaTeX strings
# print("\nLaTeX format:")
# print("f(x) =", sym.latex(fx))
# print("g(x) =", sym.latex(gx))
# print("f'(x) =", sym.latex(df))
# print("g'(x) =", sym.latex(dg))

# Manual product rule
# (fg)' = f'g + fg'
manual_product_rule_result = df * gx + fx * dg
product_rule_result_from_sympy = sym.diff(fx * gx, x)
print("\n Manual product rule ")
sym.pprint(manual_product_rule_result)

print("\n product rule via sympy")
sym.pprint(product_rule_result_from_sympy)

##########################################################################################

################### chain rule ##############################

gx = x**2 + 4*x**3 # this is x^2 +4x^3
fx = (gx)**5

print("Function for chain rule")
sym.pprint(fx)

df = sym.diff(fx)
print("Chain rule derivative is")
sym.pprint(df)

