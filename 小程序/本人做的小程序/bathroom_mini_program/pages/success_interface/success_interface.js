const { formatTime } = require('../../utils/util');
// pages/success_interface/success_interface.js
var util=require('../../utils/util');
var app=getApp();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    inputPos:'',
    time:'',
    appointment:'',
    countDownMinute:'02',
    countDownSecond:'00',
    profile:{},
  },
  bindItemTap: function() {
    wx.redirectTo({
      url: '../query_interface/query_interface'
    })
  },
  personal_func: function() {
    wx.navigateTo({
      url: '../personal_interface/personal_interface',
    })
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    var profile=app.globalData.userInfo;
    wx.getStorageInfo({
      success: (res) => {
      if (res.keys.includes('1') ){
        var value = wx.getStorageSync("1");
        // console.log(value);
        value.time=formatTime(new Date());
        this.setData({
          inputPos:value.inputPos,
          time:value.time,
          appointment:util.calTime(),
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
    //  console.log(second);
      // 天数位
      // var day = Math.floor(second / 3600 / 24);
      // var dayStr = day.toString();
      // if (dayStr.length == 1) dayStr = '0' + dayStr;
     
      // 小时位
      // var hr = Math.floor((second - day * 3600 * 24) / 3600);
      // var hrStr = hr.toString();
      // if (hrStr.length == 1) hrStr = '0' + hrStr;
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
      if (totalSecond < 0) {
       clearInterval(interval);
       wx.showToast({
        title: '时间到',
       });
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