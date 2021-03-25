import numpy as np

def get_two_state(timestep):
	state = timestep.observations['info_state'][0]

	try:
		# A does not have the ball
		pos = np.nonzero(state[0:20])[0]
		x = int(pos % 5)
		y = int(pos / 5)
		apos = np.array([x, y])
		a_status = 0
	except:
		# A has the ball
		try:
			# Game is underway and A has the ball
			pos = np.nonzero(state[20:40])[0]
			x = int(pos % 5)
			y = int(pos / 5)
			apos = np.array([x, y])
			a_status = 1
		except:
			# A scored the goal (game ended) and has the ball
			apos = np.array([5, 1.5])
			a_status = 1

	try:
		# B does not have the ball
		pos = np.nonzero(state[40:60])[0]
		x = int(pos % 5)
		y = int(pos / 5)
		bpos = np.array([x, y])
		b_status = 0
	except:
		# B has the ball
		try:
			# Game is underway and B has the ball
			pos = np.nonzero(state[60:80])[0]
			x = int(pos % 5)
			y = int(pos / 5)
			bpos = np.array([x, y])
			b_status = 1
		except:
			# B scored goal (game ended) and has the ball
			bpos = np.array([-1, 1.5])
			b_status = 1

	# Neither A nor B has the ball
	if a_status == 0 and b_status == 0:
		pos = np.nonzero(state[80:100])[0]
		x = int(pos % 5)
		y = int(pos / 5)
		opos = np.array([x, y])
	# A has the ball
	elif a_status == 1:
		opos = apos
	# B has the ball
	elif b_status == 1:
		opos = bpos

	# B's goal wrt position of A
	G1b_a = np.array([4, 1]) - apos
	G2b_a = np.array([4, 2]) - apos
	# A's goal wrt position of B
	G1a_b = np.array([0, 1]) - bpos
	G2a_b = np.array([0, 2]) - bpos
	# Ball's position wrt A
	o_a = opos - apos
	# Ball's position wrt B
	o_b = opos - bpos

	s_A = np.array([o_a, o_b, G1b_a, G2b_a, G1a_b, G2a_b]).reshape(12,)
	s_B = np.array([o_b, o_a, G1a_b, G2a_b, G1b_a, G2b_a]).reshape(12,)

	return a_status, b_status, s_A, s_B