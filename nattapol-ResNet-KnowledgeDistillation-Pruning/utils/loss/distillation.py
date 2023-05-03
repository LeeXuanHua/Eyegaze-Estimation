import torch
import torch.nn as nn
import numpy as np

class OneFeatureLoss(nn.Module):
    def __init__(self, device):
        super(OneFeatureLoss, self).__init__()
        self.COEFATLOSS = 7e-6
        self.COEFAMLOSS = 4e-4
        self.T = 0.1
        self.device = device

    def channelPool(self, feature):
        return (torch.sum(feature, (2,3)) / feature.size(2) / feature.size(3)).to(self.device)

    def spatialPool(self, feature):
        return (torch.sum(feature, 1) / feature.size(1)).to(self.device)

    def getPool(self, feature):
        return self.spatialPool(feature), self.channelPool(feature)

    def forward(self, featureStudent, featureTeacher):
        assert featureStudent != None, "Got None features"
        assert featureStudent.size() == featureTeacher.size(), "T-S features should have same dim"
        spatialStudent, channelStudent = self.getPool(featureStudent)
        spatialTeacher, channelTeacher = self.getPool(featureTeacher)

        spatialMask = featureStudent.size(2) * featureTeacher.size(3) * nn.Softmax()((spatialStudent + spatialTeacher)/ self.T).to(self.device)
        channelMask = featureTeacher.size(1) * nn.Softmax()((channelStudent+channelTeacher) / self.T).to(self.device)
        self.ATLOSS = nn.MSELoss()(spatialStudent, spatialTeacher) + nn.MSELoss()(channelStudent, channelTeacher)
        spatialMask = spatialMask.view(featureStudent.size(0), 1, featureStudent.size(2), featureStudent.size(2))
        channelMask = channelMask.view(featureStudent.size(0), featureStudent.size(1), 1, 1)
        diffMask = ((torch.square(featureStudent-featureTeacher) * spatialMask) * channelMask).to(self.device)
        self.AMLOSS = torch.sqrt(torch.sum(diffMask))
        return self.ATLOSS * self.COEFATLOSS + self.AMLOSS * self.COEFAMLOSS
    
class DistillationLoss(nn.Module):
    def __init__(self, device=torch.device("cpu"), coefAt=1, coefAm=1, coefNld=1):
        super(DistillationLoss, self).__init__()
        self.AtLoss = 0
        self.AmLoss = 0
        self.NldLoss = 0
        self.coefAt = coefAt
        self.coefAm = coefAm
        self.coefNld = coefNld
        self.T = 2000
        self.device = device
    def forward(self, outputs, gts, fStudentList, fTeacherList):
        assert len(fStudentList) == len(fTeacherList), "mismatch T-S features"
        loss = nn.MSELoss()(outputs, gts)
        ran = np.random.randint(1,100)
        for idx in range(len(fStudentList)):
            loss += (OneFeatureLoss(self.device)(fStudentList[idx], fTeacherList[idx]) * 0.05)
        return loss