var util=require('../../utils/util')
// pages/full_soon_interface/full_soon_interface.js
var app=getApp();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    inputPos:'',
    time:'',
    countDownMinute:'',
    countDownSecond:'',
    profile:{},
  },

  /**
   * 生命周期函数--监听页面加载
   */
  personal_func: function() {
    wx.navigateTo({
      url: '../personal_interface/personal_interface',
    })
  },
  backToLast: function() {
    wx.redirectTo({
      url: '../count_down_interface/count_down_interface',
    })
  },
  onLoad: function (options) {
    var profile=app.globalData.userInfo;
    wx.getStorageInfo({
      success: (res) => {
      if (res.keys.includes('2') ){
        var value = wx.getStorageSync("2");
        this.setData({
          inputPos:value.inputPos,
          time:util.formatTime(new Date()),
          profile:profile,
        });
      } 
    
  }

  })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {
    var totalSecond=(Date.parse(this.data.appointment)-Date.parse(this.data.time))/1000;
    totalSecond=120;
    var interval = setInterval(function () {
      // 秒数
      var second = totalSecond;
     var day=0;
     var hr=0;
      // 分钟位
      var min = Math.floor((second - day * 3600 *24 - hr * 3600) / 60);
      var minStr = min.toString();
      if (minStr.length == 1) minStr = '0' + minStr;
     
      // 秒位
      var sec = second - day * 3600 * 24 - hr * 3600 - min*60;
      var secStr = sec.toString();
      if (secStr.length == 1) secStr = '0' + secStr;
     
      this.setData({
      //  countDownDay: dayStr,
      //  countDownHour: hrStr,
       countDownMinute: minStr,
       countDownSecond: secStr,
      });
      totalSecond--;
      if (totalSecond<0) {
       clearInterval(interval);
       wx.showToast({
         title: '时间到',
       })
       this.setData({
        // countDownDay: '00',
        // countDownHour: '00',
        countDownMinute: '00',
        countDownSecond: '00',
       });
      }
     }.bind(this), 1000);
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