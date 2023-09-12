console.log("ToxicSense extension started !!");
function xml2string(node) {
  if (typeof XMLSerializer !== "undefined") {
    var serializer = new XMLSerializer();
    return serializer.serializeToString(node);
  } else if (node.xml) {
    return node.xml;
  }
}

var scanned_tweets = [];
var scanned_results = [];
var blurToxic = false;

function colorChanger() {
  let tweets = document.querySelectorAll("article");
  tweets.forEach(function (tweet) {
    if (window.scanned_tweets.includes(tweet)) {
      var the_tweet_no = window.scanned_tweets.indexOf(tweet);
      var resp = window.scanned_results[the_tweet_no];
      var fields = resp.split("|");
      var fakeness = fields[0];
      var toxicity = fields.slice(1, fields.length);
      //console.log("fakeness: ", fakeness)
      //console.log("toxicity: ",toxicity)
      if (toxicity.length > 0) {
        var firststring = "";
        if(blurToxic) {
            firststring = "-webkit-filter: blur(5px); -moz-filter: blur(5px); -o-filter: blur(5px); -ms-filter: blur(5px); filter: blur(5px); ";
        }
            var stringy = "";
            if(toxicity.includes('toxic')) {
              stringy = stringy.concat('#20B1E2, ')
            }if(toxicity.includes('severe_toxic')) {
              stringy = stringy.concat('#9CF4E8, ')
            }if(toxicity.includes('obscene')) {
              stringy = stringy.concat('#C76348, ')
            }if(toxicity.includes('threat')) {
              stringy = stringy.concat('#517949, ')
            }if(toxicity.includes('insult')) {
              stringy = stringy.concat('#1A0C45, ')
            }if(toxicity.includes('identity_hate')) {
              stringy = stringy.concat('#A4D265, ')
            }
            console.log("stringy: ", stringy)
            var finalstringy = stringy.slice(0,stringy.length - 2);
            finalstringy = finalstringy.concat(')');
            var template = "background-image: linear-gradient(to right, ";
            template = template.concat(finalstringy);
            template = firststring.concat(template);
            console.log("final stringy: ",finalstringy)
            console.log("template: ",template)
            tweet.setAttribute(
              "style",
              template
            );
            
      } 
    }
    // console.log("Entered");
    // let tweet_string = xml2string(tweet);
    // var text = tweet_string.replace(/<(?:.|\n)*?>/gm, "");
    // console.log(text);
    else {
      console.log("scanned_results1 :", scanned_results);
      var xhttp = new XMLHttpRequest();
      xhttp.open("POST", "http://127.0.0.1:5000/_api_call", true);
      // xhttp.send(text);
      xhttp.send(tweet.outerHTML);
      xhttp.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
          var resp = xhttp.responseText;

          window.scanned_tweets.push(tweet);
          window.scanned_results.push(resp);

          console.log(resp);
          var fields = resp.split("|");
          var fakeness = fields[0];
          var toxicity = fields.slice(1, fields.length);
          if (toxicity.length > 1) {
            var firststring = "";
        if(blurToxic) {
            firststring = "-webkit-filter: blur(5px); -moz-filter: blur(5px); -o-filter: blur(5px); -ms-filter: blur(5px); filter: blur(5px); ";
        }
            var stringy = "";
            if(toxicity.includes('toxic')) {
              stringy = stringy.concat('#20B1E2, ')
            }if(toxicity.includes('severe_toxic')) {
              stringy = stringy.concat('#9CF4E8, ')
            }if(toxicity.includes('obscene')) {
              stringy = stringy.concat('#C76348, ')
            }if(toxicity.includes('threat')) {
              stringy = stringy.concat('#517949, ')
            }if(toxicity.includes('insult')) {
              stringy = stringy.concat('#1A0C45, ')
            }if(toxicity.includes('identity_hate')) {
              stringy = stringy.concat('#A4D265, ')
            }
            console.log("stringy: ", stringy)
            var finalstringy = stringy.slice(0,stringy.length - 2);
            finalstringy = finalstringy.concat(')');
            var template = "background-image: linear-gradient(to right, ";
            template = template.concat(finalstringy);
            template = firststring.concat(template);
            console.log("final stringy: ",finalstringy)
            console.log("template: ",template)
            tweet.setAttribute(
              "style",
              template
            );
          }
        }
      };
      fake_ratio();
    }
  });
}

chrome.runtime.onMessage.addListener(
  function(request, sender) {
    blurToxic = request.blurr;
  }
);

let timer = setInterval(colorChanger, 2000);

function fake_ratio(){
  var fake=0, not_fake=0;
  var toxic=0, not_toxic=0;
  for(var i=0; i<window.scanned_results.length; i++){
    if (window.scanned_results[i].includes('not-fake')){
      not_fake += 1;
    }
    else{
      fake += 1;
    }
    if (window.scanned_results[i].split('|').length > 1){
      toxic += 1;
    }
    else{
      not_toxic += 1;
    }
  }

  // chrome.runtime.sendMessage({
  //   "total": fake + not_fake,
  //   "fake": fake,
  //   "toxic": toxic
  // });
}
