def convert2npz(input_filename, out_data_filename):
	input = open(input_filename,"r")
	output_feature = open(out_data_filename,"w")
	#output_query = open(out_query_filename,"w")
	#output_label = open(out_query_filename2,"w")

	while True:
		line = input.readline()
		if not line:
			break
		tokens = line.split(' ')
		tokens[-1] = tokens[-1].strip()
		label = tokens[0]
		qid = int(tokens[1].split(':')[1])

		#output_label.write(label + '\n')
		#output_query.write(qid + '\n')
		output_feature.write(label+' ')
		output_feature.write(qid + ' ')
		output_feature.write(' '.join(tokens[2:]) + '\n')
	
	input.close()
	output_query.close()
	output_feature.close()
	output_query2.close()

convert("set1.train.txt","yahoo.train")
convert("set1.test.txt","yahoo.test")
