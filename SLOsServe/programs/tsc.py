import torch 

model = torch.jit.load('tsc.pt')

print(model)
print('code', model.code)
print('graph', model.graph, type(model.graph))

for node in model.graph.nodes():
    print(f"Node kind: {node.kind()}")
    print(f"Inputs: {list(node.inputs())}")
    print(f"Outputs: {list(node.outputs())}")

torch.Graph
# def beam_search(initial_input):
#     beam = [initial_input]
#     beam_log_probs = torch.tensor(1.0).view(1,-1)

#     for i in range(20):
#         beam, beam_log_probs = model(beam, beam_log_probs)
    
#     return beam[0]

# initial_input = torch.arange(0, 128, dtype = torch.int32).view(1,-1)
# result = beam_search(initial_input)
# print(result)