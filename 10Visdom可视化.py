from visdom import Visdom

viz = Visdom()

viz.line([0.],[0.],win='train_loss',opts=dict(title='train loss'))
viz.line([loss.item()],[global_step],win='train_loss',update='append')
#多条曲线
viz.line([[0.,0.]],win='test',opts=dict(title='train loss&acc.',
                                        legend=['loss','acc.']))
viz.line([[test_loss,correct/len(testloader.dataset)]],
         [global_step],win='test',update='append')
#keshihua
viz.images(data.view(-1,1,28,28),win='x')
viz.text(str(pred.detach().cpu().numpy()),win='pred',
         opts=dict(title='pred'))