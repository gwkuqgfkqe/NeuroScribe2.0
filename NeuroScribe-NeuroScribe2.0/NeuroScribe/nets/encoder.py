
import torch
from torch import nn
#from sklearn import
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def transpose_to_4d_input(x):
    while len(x.shape) < 4:
        x = x.unsqueeze(-1)
    return x.permute(0, 3, 1, 2)
def get_edges(dataset):
    edges = []
    if dataset == 'bci2a':
        edges = [
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 3), (2, 7), (2, 8),
            (3, 4), (3, 9),
            (4, 5), (4, 10),
            (5, 6), (5, 11),
            (6, 12), (6, 13),
            (7, 8), (7, 14),
            (8, 9), (8, 14),
            (9, 10), (9, 15),
            (10, 11), (10, 16),
            (11, 12), (11, 17),
            (12, 13), (12, 18),
            (13, 18),
            (14, 15), (14, 19),
            (15, 16), (15, 19),
            (16, 17), (16, 20),
            (17, 18), (17, 21),
            (18, 21),
            (19, 20), (19, 22),
            (20, 21), (20, 22),
            (21, 22),
            (24, 25), (24, 29),
            (25, 26), (25, 29),
            (26, 17), (26, 30),
            (27, 28), (27, 31),
            (28, 31),
            (29, 30), (29, 32),
            (30, 31), (30, 32),
            (31, 32),
            (34, 35), (34, 39),
            (35, 36), (35, 39),
            (36, 37), (36, 30),
            (37, 18), (37, 41),
            (38, 41),
            (39, 40), (39, 42),
            (30, 41), (30, 42),
            (31, 42),
            (44, 45), (44, 49),
            (45, 46), (45, 49),
            (46, 47), (46, 50),
            (47, 48), (47, 51),
            (48, 51),
            (49, 50), (49, 52),
            (50, 51), (50, 52),
            (51, 52),
            (54, 55), (54, 59),
            (55, 56), (55, 59),
            (56, 57), (56, 60),
            (57, 58), (57, 61),
            (58, 61),
            (59, 60), (59, 62),
            (50, 61), (60, 62),
            (61, 62)
        ]
    return edges

def get_adjacency_matrix(n_electrodes, graph_strategy):
    adjacency = torch.zeros(n_electrodes, n_electrodes)
    if graph_strategy == 'CG':
        edges = get_edges('bci2a')
        for i, j in edges:
             adjacency[i - 1][j - 1] = 1
             adjacency[j - 1][i - 1] = 1
        for i in range(n_electrodes):
             adjacency[i][i] = 1
    #result = [[1.0, 0.8856935957418692, 0.9382924531313382, 0.9510956441248231, 0.941200621102631, 0.8980119661420292, 0.67755386891837, 0.7414489360100696, 0.832310229622061, 0.8407980558065943, 0.8434142297622748, 0.7605481821543035, 0.7020076535792714, 0.6076660228317878, 0.6960358622481401, 0.7028736025444705, 0.7060809553359934, 0.6470795315395058, 0.5442183403836189, 0.5774627488145575, 0.5573869148844773, 0.43318558564202764], [0.8856935957418692, 1.0, 0.9605680142740586, 0.9067552184108756, 0.8514492487775559, 0.8015836757384769, 0.8573991139046705, 0.9223754619413379, 0.9149941650677579, 0.8620357349824133, 0.8082406777284602, 0.7159004847468081, 0.6310477123933939, 0.7816147201750068, 0.7982547912227279, 0.763166502915695, 0.7206277711129229, 0.636622274901883, 0.6497066248550929, 0.6455269366062626, 0.5975201947524542, 0.49744370272257754], [0.9382924531313382, 0.9605680142740586, 1.0, 0.9693713335572016, 0.9319207262470056, 0.8597179473904232, 0.7930756233207222, 0.8781625996726792, 0.9494387603109357, 0.9262482289809723, 0.8891007773028001, 0.7726535605073778, 0.689380191328749, 0.7543853553082283, 0.8290895765954763, 0.810653882625422, 0.7817786850400303, 0.6922774965578857, 0.6689803782007248, 0.6849781135749191, 0.6429693116472717, 0.5265508916482611], [0.9510956441248231, 0.9067552184108756, 0.9693713335572016, 1.0, 0.9665313506534694, 0.912621170119102, 0.7010338825096524, 0.8080793847648122, 0.90907292026733, 0.9412278984074252, 0.9129830768828524, 0.8163751154712447, 0.7191307412328295, 0.6949890969903247, 0.787348009539575, 0.8099967543631366, 0.7950077585791087, 0.7149199240541736, 0.6362840981657091, 0.6711221921472489, 0.6426399391840009, 0.5118754992731926], [0.941200621102631, 0.8514492487775559, 0.9319207262470056, 0.9665313506534694, 1.0, 0.9633299904927018, 0.6727592860470734, 0.7610992453901407, 0.8837742010918115, 0.9215276979272019, 0.9518217841507681, 0.8844039844565915, 0.8088252570915787, 0.6687861891271389, 0.7833514667232672, 0.8125081722425979, 0.8361550712946686, 0.7845068130735006, 0.6438855217518314, 0.6921981080208925, 0.6839495794394908, 0.5378076692343091], [0.8980119661420292, 0.8015836757384769, 0.8597179473904232, 0.912621170119102, 0.9633299904927018, 1.0, 0.6215195897986228, 0.7129163315036537, 0.8125985686054977, 0.8668249973489484, 0.9259624938236742, 0.926413021903327, 0.8702175212125042, 0.6296655668399552, 0.7270096401606994, 0.7779764552972429, 0.8272709388036935, 0.8137697749894647, 0.6123928527161148, 0.6689605988769453, 0.68282995651171, 0.5280824075435587], [0.67755386891837, 0.8573991139046705, 0.7930756233207222, 0.7010338825096524, 0.6727592860470734, 0.6215195897986228, 1.0, 0.9282501408965785, 0.8419916686582899, 0.7265181142951809, 0.6849927282016048, 0.5991038707943502, 0.5512008474458009, 0.875195553808209, 0.8139695763099881, 0.7146230509340183, 0.6588151002454566, 0.5808258980205458, 0.7151837812829203, 0.6604853481306293, 0.5950328000486508, 0.5437962687296231], [0.7414489360100696, 0.9223754619413379, 0.8781625996726792, 0.8080793847648122, 0.7610992453901407, 0.7129163315036537, 0.9282501408965785, 1.0, 0.9387310167498062, 0.8485698905929555, 0.7833385165275997, 0.6941404456610847, 0.6056242057341322, 0.9352444477064429, 0.8990397040189594, 0.8266051996210636, 0.7593406532620653, 0.665839534249113, 0.7844712785614143, 0.746345720910062, 0.6796726652923336, 0.6089484838596847], [0.832310229622061, 0.9149941650677579, 0.9494387603109357, 0.90907292026733, 0.8837742010918115, 0.8125985686054977, 0.8419916686582899, 0.9387310167498062, 1.0, 0.9603979594635663, 0.9125094505207153, 0.791854615181174, 0.7025769233793158, 0.8846279246893841, 0.9488540108638793, 0.9170131959442398, 0.8668378244004667, 0.761538045220672, 0.817125633032968, 0.819115581160782, 0.7623679513325065, 0.6610224601402793], [0.8407980558065943, 0.8620357349824133, 0.9262482289809723, 0.9412278984074252, 0.9215276979272019, 0.8668249973489484, 0.7265181142951809, 0.8485698905929555, 0.9603979594635663, 1.0, 0.9559795610620745, 0.8513935527430987, 0.7444618354192121, 0.8001961536968301, 0.9073075897593174, 0.9386207825385594, 0.9046220486921135, 0.8054313882740778, 0.7828964201698698, 0.8193881840500933, 0.7804705587051501, 0.6538081127401878], [0.8434142297622748, 0.8082406777284602, 0.8891007773028001, 0.9129830768828524, 0.9518217841507681, 0.9259624938236742, 0.6849927282016048, 0.7833385165275997, 0.9125094505207153, 0.9559795610620745, 1.0, 0.9465304610673319, 0.8647154760540016, 0.7457068031569793, 0.8727772473467861, 0.9160402741297814, 0.9499708791105422, 0.9039759127164688, 0.7662598297401975, 0.8237688963865674, 0.8241296470750366, 0.670907761434035], [0.7605481821543035, 0.7159004847468081, 0.7726535605073778, 0.8163751154712447, 0.8844039844565915, 0.926413021903327, 0.5991038707943502, 0.6941404456610847, 0.791854615181174, 0.8513935527430987, 0.9465304610673319, 1.0, 0.9441541595998132, 0.6687278366972947, 0.770638887274474, 0.8398770019487839, 0.9263651785593235, 0.9526203763659719, 0.7008352015717646, 0.7711864284633487, 0.8126223786368768, 0.6428526079202], [0.7020076535792714, 0.6310477123933939, 0.689380191328749, 0.7191307412328295, 0.8088252570915787, 0.8702175212125042, 0.5512008474458009, 0.6056242057341322, 0.7025769233793158, 0.7444618354192121, 0.8647154760540016, 0.9441541595998132, 1.0, 0.5859896128748961, 0.6910190474631873, 0.7482952395693054, 0.8566100852281175, 0.9194898420263238, 0.6354285616100274, 0.7092981353271641, 0.7719788460704555, 0.6075170822898863], [0.6076660228317878, 0.7816147201750068, 0.7543853553082283, 0.6949890969903247, 0.6687861891271389, 0.6296655668399552, 0.875195553808209, 0.9352444477064429, 0.8846279246893841, 0.8001961536968301, 0.7457068031569793, 0.6687278366972947, 0.5859896128748961, 1.0, 0.9443165473658127, 0.8671793006824428, 0.7916793795260697, 0.6985794873609192, 0.9096157832911418, 0.8486058183829862, 0.7743892406895024, 0.7461446760962476], [0.6960358622481401, 0.7982547912227279, 0.8290895765954763, 0.787348009539575, 0.7833514667232672, 0.7270096401606994, 0.8139695763099881, 0.8990397040189594, 0.9488540108638793, 0.9073075897593174, 0.8727772473467861, 0.770638887274474, 0.6910190474631873, 0.9443165473658127, 1.0, 0.9643921621106067, 0.9065127625841363, 0.8025541297375008, 0.9409338889616375, 0.9300998560431049, 0.8674243494237269, 0.8012119341441022], [0.7028736025444705, 0.763166502915695, 0.810653882625422, 0.8099967543631366, 0.8125081722425979, 0.7779764552972429, 0.7146230509340183, 0.8266051996210636, 0.9170131959442398, 0.9386207825385594, 0.9160402741297814, 0.8398770019487839, 0.7482952395693054, 0.8671793006824428, 0.9643921621106067, 1.0, 0.9618720904664214, 0.8695284545843612, 0.9193532438384807, 0.9522375002570201, 0.9136108015508483, 0.8161124661696677], [0.7060809553359934, 0.7206277711129229, 0.7817786850400303, 0.7950077585791087, 0.8361550712946686, 0.8272709388036935, 0.6588151002454566, 0.7593406532620653, 0.8668378244004667, 0.9046220486921135, 0.9499708791105422, 0.9263651785593235, 0.8566100852281175, 0.7916793795260697, 0.9065127625841363, 0.9618720904664214, 1.0, 0.963455287634792, 0.8695896920754724, 0.9300457622839666, 0.9426630318968814, 0.807375554141569], [0.6470795315395058, 0.636622274901883, 0.6922774965578857, 0.7149199240541736, 0.7845068130735006, 0.8137697749894647, 0.5808258980205458, 0.665839534249113, 0.761538045220672, 0.8054313882740778, 0.9039759127164688, 0.9526203763659719, 0.9194898420263238, 0.6985794873609192, 0.8025541297375008, 0.8695284545843612, 0.963455287634792, 1.0, 0.7846798783142426, 0.8582674613338758, 0.9157519550106691, 0.7600755412690241], [0.5442183403836189, 0.6497066248550929, 0.6689803782007248, 0.6362840981657091, 0.6438855217518314, 0.6123928527161148, 0.7151837812829203, 0.7844712785614143, 0.817125633032968, 0.7828964201698698, 0.7662598297401975, 0.7008352015717646, 0.6354285616100274, 0.9096157832911418, 0.9409338889616375, 0.9193532438384807, 0.8695896920754724, 0.7846798783142426, 1.0, 0.9732627741206104, 0.9197722504602858, 0.9281928054983251], [0.5774627488145575, 0.6455269366062626, 0.6849781135749191, 0.6711221921472489, 0.6921981080208925, 0.6689605988769453, 0.6604853481306293, 0.746345720910062, 0.819115581160782, 0.8193881840500933, 0.8237688963865674, 0.7711864284633487, 0.7092981353271641, 0.8486058183829862, 0.9300998560431049, 0.9522375002570201, 0.9300457622839666, 0.8582674613338758, 0.9732627741206104, 1.0, 0.970664882253634, 0.9362930912897099], [0.5573869148844773, 0.5975201947524542, 0.6429693116472717, 0.6426399391840009, 0.6839495794394908, 0.68282995651171, 0.5950328000486508, 0.6796726652923336, 0.7623679513325065, 0.7804705587051501, 0.8241296470750366, 0.8126223786368768, 0.7719788460704555, 0.7743892406895024, 0.8674243494237269, 0.9136108015508483, 0.9426630318968814, 0.9157519550106691, 0.9197722504602858, 0.970664882253634, 1.0, 0.9302976863807262], [0.43318558564202764, 0.49744370272257754, 0.5265508916482611, 0.5118754992731926, 0.5378076692343091, 0.5280824075435587, 0.5437962687296231, 0.6089484838596847, 0.6610224601402793, 0.6538081127401878, 0.670907761434035, 0.6428526079202, 0.6075170822898863, 0.7461446760962476, 0.8012119341441022, 0.8161124661696677, 0.807375554141569, 0.7600755412690241, 0.9281928054983251, 0.9362930912897099, 0.9302976863807262, 1.0]]

    #adjacency = torch.tensor(result)
        #adjacency = normalize_adjacency_matrix(adjacency)
    return adjacency




class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        # Set the maximum norm
        self.max_norm = max_norm
        # Calls the initialization method of the parent class Conv2d
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        # Re-normalize the weight data
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        # Calls the parent Conv2dWithConstraint forward method and returns the result
        return super(Conv2dWithConstraint, self).forward(x)




class SpectralAttentionModule(nn.Module):
    def __init__(self, num_channels):
        super(SpectralAttentionModule, self).__init__()
        self.num_channels = num_channels

    def forward(self, x):
        # FFT transforms into the frequency domain
        x_fft = torch.fft.rfft(x, dim=-1).to(device)

        # Get the frequency dimension after FFT transformation
        num_freqs = x_fft.shape[-1]
        num_featuregra=x.shape[1]

        # Initialize the attention parameter, note that it is initialized in the forward method
        # Use normal distribution to initialize attention parameters
        query = nn.Parameter(torch.randn(num_featuregra,self.num_channels, num_freqs)).to(device)
        key = nn.Parameter(torch.randn(num_featuregra,self.num_channels, num_freqs)).to(device)
        value = nn.Parameter(torch.randn(num_featuregra,self.num_channels, num_freqs)).to(device)

        nn.init.normal_(query)
        nn.init.normal_(key)
        nn.init.normal_(value)

        # Calculate attention weight
        batch_size, _, _,_ = x_fft.size()
        query = query.unsqueeze(0).repeat(batch_size, 1, 1,1)
        key = key.unsqueeze(0).repeat(batch_size, 1, 1,1)
        value = value.unsqueeze(0).repeat(batch_size, 1, 1,1)

        attn_logits = torch.matmul(query, key.transpose(-1, -2)).to(device)
        attn_weights = F.softmax(attn_logits, dim=-1).to(device)

        # Attention weighting
        attn_output = torch.matmul(attn_weights, value) * x_fft

        # Inverse FFT transform back to time domain
        x_ifft = torch.fft.irfft(attn_output, dim=-1).to(device)

        return x_ifft


class TemporalAttentionModule(nn.Module):
    def __init__(self, num_channels):
        super(TemporalAttentionModule, self).__init__()
        self.num_channels = num_channels

    def forward(self, x):

        num_freqs = x.shape[-1]
        num_featuregra=x.shape[1]


        query = nn.Parameter(torch.randn(num_featuregra,self.num_channels, num_freqs)).to(device)
        key = nn.Parameter(torch.randn(num_featuregra,self.num_channels, num_freqs)).to(device)
        value = nn.Parameter(torch.randn(num_featuregra,self.num_channels, num_freqs)).to(device)

        nn.init.normal_(query)
        nn.init.normal_(key)
        nn.init.normal_(value)


        batch_size, _, _,_ = x.size()
        query = query.unsqueeze(0).repeat(batch_size, 1, 1,1)
        key = key.unsqueeze(0).repeat(batch_size, 1, 1,1)
        value = value.unsqueeze(0).repeat(batch_size, 1, 1,1)

        attn_logits = torch.matmul(query, key.transpose(-1, -2)).to(device)
        attn_weights = F.softmax(attn_logits, dim=-1).to(device)


        x = torch.matmul(attn_weights, value) * x



        return x


class GraphTemporalConvolution(nn.Module):
    def __init__(self, adjacency, in_channels, out_channels, kernel_length):
        # Call the parent class initialization method
        super(GraphTemporalConvolution, self).__init__()
        # Enter the number of channels
        self.in_channels = in_channels
        # Number of output channels
        self.out_channels = out_channels
        # Register a buffer to store the adjacency matrix
        self.register_buffer('adjacency', adjacency)
        # Initializes a learnable parameter importance with the size (number of input channels, adjacency matrix size, adjacency matrix size)
        self.importance = nn.Parameter(torch.randn(in_channels, self.adjacency.size()[0], self.adjacency.size()[0]))
        # Initializes a 2D convolution layer with the number of input channels as the number of input channels, the number of output channels as the number of output channels, the size of the convolution kernel as (1, kernel_length), the step size as 1, no bias items used, and the fill mode as 'same'.
        self.temporal_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(1, kernel_length), stride=1, bias=False, padding='same')

    def forward(self, x):
        # Multiply the adjacency matrix with the importance matrix, and then multiply the input matrix x
        x = torch.matmul(torch.mul(self.adjacency, self.importance), x)
        # Process the result through the time convolution layer
        x = self.temporal_conv(x)
        return x








class GCNEEGNet(nn.Module):
    """
    Initialize the GCNEEGNet class.

    Args:
    n_channels (int): Number of channels to enter data.
    n_classes (int): Number of output classes.
    input_window_size (int): The size of the window in which data is entered.
    F1 (int, optional): Number of filters for the first convolution layer. The default value is 8.
    D (int, optional): The number of groups of separable convolution in the second convolution layer. The default is 2.
    F2 (int, optional): Number of filters for the second convolution layer. The default value is 16.
    kernel_length (int, optional): length of the convolution kernel. The default value is 64.
    drop_p (float, optional): Dropout probability of the dropout layer. The default is 0.5.

    Returns:
    None

    """
    def __init__(self, n_channels, n_classes, input_window_size,
                 F1=8, D=2, F2=16, kernel_length=64, drop_p=0.25):
        super(GCNEEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_p = drop_p
        self.input_windows_size = input_window_size
        adjacency = get_adjacency_matrix(self.n_channels, 'CG')
        self.frenquencyatten = SpectralAttentionModule(self.n_channels)
        self.temporalatten=TemporalAttentionModule(self.n_channels)
        self.block_temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_length),stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            GraphTemporalConvolution(adjacency, self.F1, self.F1, 8),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
        )
        self.block_spacial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_channels, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_p)
        )
        self.block_separable_conv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_p),
            nn.Flatten()
        )
        block_conv = nn.Sequential(
            self.block_temporal_conv,
            self.block_spacial_conv,
            self.block_separable_conv
        )
        out = block_conv(torch.ones((1, 1, self.n_channels, self.input_windows_size), dtype=torch.float32))
        self.block_classifier = nn.Sequential(
            nn.Linear(out.cpu().data.numpy().shape[1], self.n_classes),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):#(64,22,1125)
        # x=self.temporalatten(x)
        
        
        x = transpose_to_4d_input(x)#(64,1,22,1125)
        x=self.block_temporal_conv(x)
        #x=self.temporalatten(x)
        #x=self.frenquencyatten(x)
        #x = self.block_temporal_conv(x)#(64,8,22,1125)
        x = self.block_spacial_conv(x)#(64,16,1,281)
        x = self.block_separable_conv(x)#(64,560)
        x = self.block_classifier(x)#（64，4）
        return x




from braindecode.models import EEGConformer,ATCNet,EEGNetv4


class MergedModel(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_size, kernel_length=64):
        super(MergedModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernel_length = kernel_length
        # self.batch_size=batch_size
        self.input_windoe_size = input_window_size
        # self.frenquencyatten = SpectralAttentionModule(self.n_channels)
        # self.lstm = nn.LSTM(62, 62, 3, batch_first=True)
        self.gcnEEGNet = GCNEEGNet(n_channels=self.n_channels, n_classes=self.n_classes,
                                   input_window_size=self.input_windoe_size,kernel_length=64)


    def forward(self, x):


        gcn_output = self.gcnEEGNet(x)#(64,4)


        return gcn_output


