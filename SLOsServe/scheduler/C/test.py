# import promax
# import time

# # Create Request objects

# reqs = [promax.Request(True, 8, 1, 1, 3, 0),
#         promax.Request(True, 12, 2, 20, 3, 0),
#         promax.Request(True, 2, 2, 10, 3, 0),
#         promax.Request(True, 2, 2, 1, 3, 0),
#         promax.Request(False, 0, 0, 1, 3, 0),
#         promax.Request(False, 0, 0, 1, 3, 0),
#         promax.Request(False, 4, 2, 1, 3, 0)]



# reqs = sorted(reqs, key = lambda x: x.ddl)

# # Create a PromaxScheduler instance
# tpots = [4.0]
# hardware_params = [0.0, 4, 1., 0.0]
# scheduler = promax.PromaxScheduler(tpots, hardware_params, 0.8, 6)

# # Call schedule
# M = 100
# current_time = 0.0
# start = time.perf_counter()
# result = scheduler.schedule(reqs, M, current_time)

# print("Schedule result:", result)
# print('elasped time: ', time.perf_counter() - start)


import promax

# Create Request objects
reqs = [promax.Request(id=1, is_new_req=True, ddl=8, input_length=1, profit=1, mem=3, tpot_idx =0),
        promax.Request(2, True, 12, 2, 20, 3, 0),
        promax.Request(3, True, 2, 2, 10, 3, 0),
        promax.Request(4, True, 2, 2, 1, 3, 0),
        promax.Request(5, False, 0, 0, 1, 3, 0),
        promax.Request(6, False, 0, 0, 1, 3, 0),
        promax.Request(7, False, 4, 2, 1, 3, 0)]


# Print Request details
print(reqs)

# Create a PromaxScheduler instance
tpots = [4.0]
hardware_params = [0.0, 0.0, 4, 1., 0.0, 0.0]
scheduler = promax.PromaxScheduler()
scheduler.set_sd_planner(tpots, hardware_params, False, 0.8, 10, False)

# Call schedule
M = 100
current_time = 0.0
is_feasible, accepted_ids, batch_schedules = scheduler.schedule(reqs, M, current_time, True)

print('feasible:', is_feasible)

print('acc_ids', accepted_ids)

print(reqs)

# Print the schedule results
print("Schedule result:")
for batch_sch in batch_schedules:
    print(batch_sch)
    for req_batch_sch in batch_sch.req_batches:
        print("  ", req_batch_sch)
