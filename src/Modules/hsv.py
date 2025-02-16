import cv2
import numpy as np

class HSVImage:
    def __init__(self, image):
        self._original_image = image.copy()  # プライベート変数として定義
        self._hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def get_mask(self, hsv_lower=None, hsv_upper=None):
        """
        HSVの範囲に基づいてマスクを生成

        Args:
            hsv_lower: HSV下限値 (tuple), デフォルト=(1, 0, 0)
            hsv_upper: HSV上限値 (tuple), デフォルト=(30, 255, 255)

        Returns:
            生成されたマスク画像
        """
        # デフォルト値の設定
        hsv_lower = np.array(hsv_lower or (1, 0, 0))
        hsv_upper = np.array(hsv_upper or (30, 255, 255))

        # マスクの生成
        self.mask = cv2.inRange(self.hsv_image, hsv_lower, hsv_upper)
        return self.mask

    def make_masked_image(self, mask=None):
        """
        マスクを適用した画像を生成

        Args:
            mask: 適用するマスク（Noneの場合は直前に生成したマスクを使用）

        Returns:
            マスク適用後の画像
        """
        if mask is None:
            if self.mask is None:
                raise ValueError("マスクが生成されていません。get_mask()を先に実行してください。")
            mask = self.mask

        self.masked_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
        return self.masked_image

    def reset(self):
        """
        画像を初期状態にリセット
        """
        self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        self.mask = None
        self.masked_image = None


    @property
    def original(self):
        """
        元画像を取得するプロパティ
        読み取り専用でアクセス可能
        """
        return self._original_image

    @property
    def hsv(self):
        """
        HSV画像を取得するプロパティ
        読み取り専用でアクセス可能
        """
        return self._hsv_image

    # セッターを定義する場合
    @hsv.setter
    def hsv(self, new_hsv_image):
        """
        HSV画像を設定するプロパティ
        値の検証などが可能
        """
        if new_hsv_image.shape != self._hsv_image.shape:
            raise ValueError("新しい画像のサイズが異なります")
        self._hsv_image = new_hsv_image

    @property
    def shape(self):
        self.shape = self._original_image.shape
        return self.shape
