import e2e_runner as e2

SELECTED_REL_CATS = [5,11,36,37]

def keep_entity(image_entities):
	return [y for y in image_entities if y ['rel_cat'] in SELECTED_REL_CATS]

def comb_dataloader(dataloader):
	data = dataloader.dataset._data
	trimmed_data = [keep_entity(image_entities) for image_entities in data]
	dataloader.dataset._data = [image_entities for image_entities in trimmed_data if len(image_entities) > 0]
	print('combed dataset len %5d %5d' % (len(dataloader), len(dataloader.dataset)))

r = e2.Runner()
r.setup()

comb_dataloader(r.trainloader)
for testloader in r.testloaders:
	comb_dataloader(testloader)

r.train()
