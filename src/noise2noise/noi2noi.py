import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread,imsave
import torch,os,time
from glob import glob
from unet import Unet # unet.pyからインポート
nn = torch.nn
mse = nn.MSELoss() # 損失関数

# noise2noiseのクラス
class Noi2noi:
    def __init__(self,hozon_folder,gakushuuritsu=1e-3,gpu=1,kataru=1):
        '''
        hozon_folder: パラメータや生成された画像を保存するフォルダ
        gakushuuritsu: 学習率
        gpu: GPUを使うか
        kataru: 学習している時、検証の結果をprintするか
        '''

        self.gakushuuritsu = gakushuuritsu
        self.kataru = kataru
        self.net = Unet(cn=3) # ネット
        self.opt = torch.optim.Adam(self.net.parameters(),lr=gakushuuritsu) # オプティマイザ
        if(gpu):
            self.dev = torch.device('mps')
            self.net = self.net.to(self.dev)
        else:
            self.dev = torch.device('cpu')

        self.hozon_folder = hozon_folder
        if(not os.path.exists(hozon_folder)):
            os.mkdir(hozon_folder) # 保存するフォルダがまだ存在していなければ作る

        netparam_file = os.path.join(hozon_folder,'netparam.pkl')
        # すでに学習して重みとか保存した場合、読み込んで使う
        if(os.path.exists(netparam_file)):
            s = torch.load(netparam_file)
            self.net.load_state_dict(s['w']) # ネットの重み
            self.opt.load_state_dict(s['o']) # オプティマイザの状態
            self.mounankai = s['n'] # もう何回学習した
            self.sonshitsu = s['l'] # 毎回の損失を納めたリスト
        else:
            self.mounankai = 0
            self.sonshitsu = [],[]


    def gakushuu(self,dalo,n_kurikaeshi,kenshou,kenshou_hindo=1,matsu=10,yarinaosu=0):
        '''
        dalo: データローダ
        n_kurikaeshi: 何回繰り返して学習するか
        kenshou: 検証のためのオブジェクト
        kenshou_hindo: 一回学習に何回検証を行うか
        matsu: 何回が損失が下らなければ学習は終了
        yarinaosu: 以前学習した重みを使わずに学習を最初からやり直すか
        '''

        # やり直す場合
        if(yarinaosu):
            self.net = Unet(cn=3).to(self.dev)
            self.opt = torch.optim.Adam(self.net.parameters(),lr=self.gakushuuritsu)
            self.mounankai = 0
            self.sonshitsu = [],[]

        t0 = time.time()
        saitei_sonshitsu = np.inf # 一番低い損失
        susundenai = 0
        for kaime in range(self.mounankai,self.mounankai+n_kurikaeshi):
            for i_batch,(x,y) in enumerate(dalo):
                z = self.net(x.to(self.dev)) # 画像を生成する
                sonshitsu = mse(z,y.to(self.dev)) # 目標画像と比べて損失を計算する
                self.opt.zero_grad() # 勾配リセット
                sonshitsu.backward() # 誤差逆伝搬
                self.opt.step() # 重み更新

                # 検証
                if((i_batch+1)%int(np.ceil(len(dalo)/kenshou_hindo))==0 or i_batch==len(dalo)-1):
                    sonshitsu = kenshou(self,kaime,i_batch)
                    if(sonshitsu<saitei_sonshitsu): # 損失が前にり低くなった場合
                        susundenai = 0
                        saitei_sonshitsu = sonshitsu
                    else: # 低くなってない場合
                        susundenai += 1
                        if(self.kataru):
                            print('もう%d回進んでない'%susundenai,end=' ')
                    if(self.kataru):
                        print('%.2f分過ぎた'%((time.time()-t0)/60))
            # 重みなどを保存する
            sd = dict(w=self.net.state_dict(),o=self.opt.state_dict(),n=kaime+1,l=self.sonshitsu)
            torch.save(sd,os.path.join(self.hozon_folder,'netparam.pkl'))
            if(susundenai>=matsu):
                break # 何回も損失が下がっていない場合、学習終了

    def __call__(self,x,n_batch=4):
        x = torch.Tensor(x)
        y = []
        for i in range(0,len(x),n_batch):
            y.append(self.net(x[i:i+n_batch].to(self.dev)).detach().cpu())
        return torch.cat(y).numpy()



# 画像のデータローダ
class Gazoudata:
    def __init__(self,folder,n_batch=4,n_tsukau=None):
        '''
        folder: 訓練データを納めたフォルダ
        n_batch: バッチサイズ
        n_tsukau: 一枚の元の画像毎に、ノイズ画像を最大何枚使うか
        '''

        self.n_batch = n_batch
        self.file = []
        for fo in sorted(glob(folder+'/*')):
            ff = glob(fo+'/*')
            if(n_tsukau):
                ff = ff[:n_tsukau]
            self.file.append(ff)

    # for が始めた時にランダムで画像を2枚ずつ組み合わせて並べる
    def __iter__(self):
        self.item = []
        for fo in self.file:
            i_rand = np.random.permutation(len(fo))
            for i in range(0,len(fo),2):
                self.item.append((fo[i_rand[i]],fo[i_rand[i+1]]))
        self.i_iter = 0
        self.i_rand = np.random.permutation(len(self.item))
        self.len = int(np.ceil(len(self.item)/self.n_batch))
        return self

    # 毎回のバッチサイズにバッチサイズと同じ数の画像を渡す
    def __next__(self):
        if(self.i_iter>=len(self.item)):
            raise StopIteration
        x = [] # 入力の画像
        y = [] # 目標の画像
        for i in self.i_rand[self.i_iter:self.i_iter+self.n_batch]:
            x.append(imread(self.item[i][0]))
            y.append(imread(self.item[i][1]))
        self.i_iter += self.n_batch
        x = torch.Tensor(np.stack(x).transpose(0,3,1,2)/255.)
        y = torch.Tensor(np.stack(y).transpose(0,3,1,2)/255.)
        return x,y

    def __len__(self):
        return self.len



# 検証を行うオブジェクト
class Kenshou:
    def __init__(self,kunren_folder,kenshou_folder,n_kenshou,n_batch=8,gazou_kaku=1,n_kaku=5,graph_kaku=1):
        '''
        kunren_folder: 訓練データを納めたフォルダ
        kenshou_folder: 検証データを納めたフォルダ
        n_kenshou: 何枚検証に使うか
        n_batch: 検証の時のバッチサイズ
        gazou_kaku: 検証に使われた画像を書くか
        n_kaku: 何枚書くか
        graph_kaku: グラフを書くか
        '''

        gg = sorted(glob(kunren_folder+'/*/*.jpg'))[:n_kenshou]
        self.x_kunren = torch.Tensor(np.stack([imread(g) for g in gg]).transpose(0,3,1,2)/255.)
        gg = sorted(glob(kunren_folder+'/*/*.jpg'))[:n_kenshou]
        self.y_kunren = torch.Tensor(np.stack([imread(g) for g in gg]).transpose(0,3,1,2)/255.)
        gg = sorted(glob(kenshou_folder+'/*/*.jpg'))[:n_kenshou]
        self.x_kenshou = torch.Tensor(np.stack([imread(g) for g in gg]).transpose(0,3,1,2)/255.)
        gg = sorted(glob(kenshou_folder+'/*/*.jpg'))[:n_kenshou]
        self.y_kenshou = torch.Tensor(np.stack([imread(g) for g in gg]).transpose(0,3,1,2)/255.)

        self.n_kenshou = n_kenshou
        self.n_batch = n_batch
        self.gazou_kaku = gazou_kaku
        self.n_kaku = n_kaku
        self.graph_kaku = graph_kaku

    def __call__(self,model,kaime,i_batch):
        # 訓練データと検証データに対する損失を計算して格納する
        z_kunren = torch.cat([model.net(self.x_kunren[i:i+self.n_batch].to(model.dev)).cpu() for i in range(0,self.n_kenshou,self.n_batch)])
        sonshitsu_kunren = mse(z_kunren,self.y_kunren).item()
        model.sonshitsu[0].append(sonshitsu_kunren)

        z_kenshou = torch.cat([model.net(self.x_kenshou[i:i+self.n_batch].to(model.dev)).cpu() for i in range(0,self.n_kenshou,self.n_batch)])
        sonshitsu_kenshou = mse(z_kenshou,self.y_kenshou).item()
        model.sonshitsu[1].append(sonshitsu_kenshou)

        # 生成された画像を保存する
        if(self.gazou_kaku):
            img_kunren = z_kunren.detach()[:self.n_kaku].cpu().numpy().transpose(0,2,3,1)
            img_kenshou = z_kenshou.detach()[:self.n_kaku].cpu().numpy().transpose(0,2,3,1)
            img = np.stack(list(img_kunren)+list(img_kenshou)).reshape(2,-1,256,256,3)
            img = np.hstack(np.hstack(img))
            img = (np.clip(img,0,1)*256).astype(np.uint8)
            imsave(os.path.join(model.hozon_folder,'%03d_%04d.jpg'%(kaime+1,i_batch+1)),img)

        # グラフを書く
        if(self.graph_kaku):
            plt.gca(ylabel='MSE')
            plt.plot(model.sonshitsu[0])
            plt.plot(model.sonshitsu[1])
            plt.legend([u'訓練',u'検証'],prop={'family':'TakaoPGothic','size':17})
            mi = np.argmin(model.sonshitsu[1])
            m = model.sonshitsu[1][mi]
            arrowprops = {'arrowstyle':'->','color':'r'}
            plt.annotate('%.5f'%m,[mi,m],[mi*0.9,m*1.01],arrowprops=arrowprops)
            plt.tight_layout()
            plt.savefig(os.path.join(model.hozon_folder,'graph.jpg'))
            plt.close()

        if(model.kataru):
            print('%d回目の%d 損失=[%.8f,%.8f]'%(kaime+1,i_batch+1,sonshitsu_kunren,sonshitsu_kenshou))

        return sonshitsu_kenshou



# ここで設定
kunren_folder = './train_data' # 訓練のノイズ画像を納めた場所
kenshou_folder = './valid_data' # 検証のノイズ画像を納めた場所
hozon_folder = 'noi2noi' # 結果を保存する場所
n_batch = 4 # バッチサイズ
n_tsukau = 100 # 訓練の画像毎に最大何枚ノイズ画像を使うか
n_kenshou = 20 # 検証の画像毎に最大何枚ノイズ画像を使うか
n_batch_kenshou = 8 # 検証する時のバッチサイズ
gazou_kaku = 1 # 検証の時に画像を書くか
n_kaku = 5 # 検証の時に何枚画像を書くか
graph_kaku = 1 # 検証のグラフを書くか
n_kurikaeshi = 100 # 損失が結束しない場合最大何回まで続くか
kenshou_hindo = 6 # 毎回の訓練に何回検証をするか
matsu = 10 # 何回損失が下がらないと学習が終わるか
yarinaosu = 0 # 最初から学習をやり直すか
gakushuuritsu = 0.001 # 学習率
gpu = 1 # GPUを使うか



# 実行
n2n = Noi2noi(hozon_folder,gakushuuritsu,gpu) # noise2noiseモデル
dalo = Gazoudata(kunren_folder,n_batch,n_tsukau) # データローダ
kenshou = Kenshou(kunren_folder,kenshou_folder,n_kenshou,n_batch_kenshou,gazou_kaku,n_kaku,graph_kaku) # 検証オブジェクト
n2n.gakushuu(dalo,n_kurikaeshi,kenshou,kenshou_hindo,matsu,yarinaosu) # 学習開始
