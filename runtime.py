from train import Trainer

dataset_path = '/home/litongxin'
lr = 0.0002
vis_screen = 'gan'
save_path = './result'
l1_coef = 50
l2_coef = 100
batch_size = 64
num_workers = 4
epochs = 200

trainer = Trainer(dataset_path, lr, vis_screen, save_path, l1_coef, l2_coef, batch_size, num_workers, epochs)
trainer.test()
