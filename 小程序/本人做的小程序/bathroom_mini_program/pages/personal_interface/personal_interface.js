// pages/personal_interface/personal_interface.js
var app=getApp();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    focus:false,
    inputName: '',
    inputPhone:'',
    inputSex:'',
    inputNum:'',
    inputPos:'',
    profile:{},
  },
  bindname: function(e) {
    this.setData({
      inputName:e.detail.value
    })

  },
  bindtele: function(e) {
    this.setData({
      inputPhone:e.detail.value
    }
    )
  },
  bindsex: function(e) {
    this.setData({
      inputSex:e.detail.value
    }
    )
  },
  bindnum: function(e) {
    this.setData({
      inputNum:e.detail.value
    }
    )
  },
  bindpos: function(e) {
    this.setData({
      inputPos:e.detail.value
    }
    )
  },
  sureInfo: function(e) {
    try {
      wx.setStorageSync('0', this.data)
    }catch(e) {console.log(e)}
    wx.navigateTo({
      url: '../query_interface/query_interface',
    })
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    var profile=app.globalData.userInfo;
    console.log(profile);
    wx.getStorageInfo({
      success: (res) => {
        // console.log(res.keys);
      if (res.keys.includes('0') ){
        // console.log(res.keys.includes('0'));
        var value = wx.getStorageSync("0");
        // console.log(value);
        // console.log(this.data);
        this.setData({
          inputName: value.inputName,
          inputPhone:value.inputPhone,
          inputSex:value.inputSex,
          inputNum:value.inputNum,
          inputPos:value.inputPos,
          profile:profile,
        });
        // console.log(this.data);
        // if (value ) {
        //   this.globalData.inputName=value.inputName;
        //   this.globalData.inputPhone=value.inputPhone;
        //   this.globalData.inputSex=value.inputSex;
        //   this.globalData.inputNum=value.inputNum;
        //   this.globalData.inputPos=value.inputPos;
        // }
      } 
    
  }

  })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {
    
  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {
 
},

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})