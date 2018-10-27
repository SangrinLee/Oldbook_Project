# for i in range(100):
# 	print(i/100)

# for i in range(100):
	# print("=COUNTIFS(I:I, \"correct\", L:L, \">" + str(i/100) + "\")")
# for i in range(100):
	# print("=COUNTIFS(I:I, \"incorrect\", L:L, \">" + str(i/100) + "\")")
# for i in range(100):
	# print("=COUNTIFS(I:I, \"correct\", L:L, \"<" + str(i/100) + "\")")

for i in range(100):
	print("=COUNTIFS(I:I, \"incorrect\", L:L, \"<" + str(i/100) + "\")")