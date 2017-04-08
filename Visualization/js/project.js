
    //Reading different csv files required for the visualization  
    d3.json("/datafiles/countries.topo.json", function(error, topo) { //file for plotting the map
      d3.csv("/datafiles/ANN.csv", function(error,ANN){ //Predicted output on test data from Neural Network
        d3.csv("/datafiles/LogisticRegression.csv", function(error,LR){ //Predicted output on test data from Logistic regression
          d3.csv("/datafiles/XGBoost.csv", function(error,XG){ //Predicted output on test data from XGBoost
            d3.csv("/datafiles/Accuracy.csv", function(error,ACC){ //Accuracy of different files
            

            //Creating option box for selection of different model outputs
            var Options = ["ANN","Logistic Regression", "XGBoost"];
            d3.select('#map')
              .append('text')
              .attr("id","smtag")
              .attr("text-anchor", "end")
              .text("Select Model")
              .style({
                  "font-family": "Calibri",
                  "font-size":"14px",
                  "font-weight":"bold"
              });

            var select = d3.select('#map')
              .append('select')
              .attr("id","Models")
                .attr('class','select')
                .on('change',onchange)

            var options = select
              .selectAll('option')
              .data(Options).enter()
              .append('option')
                .text(function (d) { return d; });

            var selectValue = "ANN";    
            function onchange() {
              selectValue = d3.select('select').property('value');
              update();
            };

            //defining width and height
            var m_width = $("#map").width(),
                width = 700,
                height = 320,
                country,
                state;

            //variable to transform the map i.e place the country in center    
            var projection = d3.geo.mercator()
                .scale(300)
                .translate([width / 2, height / 1.5]);

            var path = d3.geo.path()
                .projection(projection);

            //defining svg    
            var svg = d3.select("#map").append("svg")
                .attr("preserveAspectRatio", "xMidYMid")
                .attr("viewBox", "0 0 " + width + " " + height)
                .attr("width", m_width)
                .attr("height", (m_width * height / width)*0.8);

            svg.append("rect")
                .attr("class", "background")
                .attr("width", width)
                .attr("height", height);

            //Plotting base map of China    
            var g = svg.append("g"); 
              g.append("g")
                .attr("id", "countries")
                .selectAll("path")
                .data(topojson.feature(topo, topo.objects.countries).features)
                .enter()
                .append("path")
                .attr("id", function(d) { return d.id; })
                .attr("d", path);

            xyz = get_xyz(topojson.feature(topo, topo.objects.countries).features[0]);

            //creating checkbox for different categories
            var checkBox1 = new d3CheckBox("Male","M"),
                checkBox2 = new d3CheckBox("22-","M"),
                checkBox3 = new d3CheckBox("23-26","M"),
                checkBox4 = new d3CheckBox("27-28","M"),
                checkBox5 = new d3CheckBox("29-31","M"),
                checkBox6 = new d3CheckBox("32-38","M"),
                checkBox7 = new d3CheckBox("39+","M"),
                checkBox8 = new d3CheckBox("Female","F"),
                checkBox9 = new d3CheckBox("23-","F"),
                checkBox10 = new d3CheckBox("24-26","F"),
                checkBox11 = new d3CheckBox("27-28","F"),
                checkBox12 = new d3CheckBox("29-32","F"),
                checkBox13 = new d3CheckBox("33-42","F"),
                checkBox14 = new d3CheckBox("43+","F");  
            
            //Function to update the visualizaton on change in selection    
            var update = function () {

                    var c1 = checkBox1.checked(),
                        c8 = checkBox8.checked(),
                        c2,c3,c4,c5,c6,c7,c9,c10,c11,c12,c13,c14;
                      c2 = checkBox2.checked();
                      c3 = checkBox3.checked();
                      c4 = checkBox4.checked();
                      c5 = checkBox5.checked();
                      c6 = checkBox6.checked();
                      c7 = checkBox7.checked();
                      c9 = checkBox9.checked();
                      c10 = checkBox10.checked();
                      c11 = checkBox11.checked();
                      c12 = checkBox12.checked();
                      c13 = checkBox13.checked();
                      c14 = checkBox14.checked();

                    if(c1==false&&(c2==true||c3==true||c4==true||c4==true||c6==true||c7==true)){
                      checkBox2.unmark();
                      checkBox3.unmark();
                      checkBox4.unmark();
                      checkBox5.unmark();
                      checkBox6.unmark();
                      checkBox7.unmark();
                      
                    }  

                    if(c8==false&&(c9==true||c10==true||c11==true||c12==true||c13==true||c14==true)){
                      checkBox9.unmark();
                      checkBox10.unmark();
                      checkBox11.unmark();
                      checkBox12.unmark();
                      checkBox13.unmark();
                      checkBox14.unmark();
                    }

                    if(c1==true&&c2==false&&c3==false&&c4==false&&c4==false&&c6==false&&c7==false){
                      checkBox2.marked();
                      checkBox3.marked();
                      checkBox4.marked();
                      checkBox5.marked();
                      checkBox6.marked();
                      checkBox7.marked();
                    }

                    if(c8==true&&c9==false&&c10==false&&c11==false&&c12==false&&c13==false&&c14==false){
                      checkBox9.marked();
                      checkBox10.marked();
                      checkBox11.marked();
                      checkBox12.marked();
                      checkBox13.marked();
                      checkBox14.marked();
                    }
                      
                      c1 = checkBox1.checked();
                      c2 = checkBox2.checked();
                      c3 = checkBox3.checked();
                      c4 = checkBox4.checked();
                      c5 = checkBox5.checked();
                      c6 = checkBox6.checked();
                      c7 = checkBox7.checked();
                      c8 = checkBox8.checked();
                      c9 = checkBox9.checked();
                      c10 = checkBox10.checked();
                      c11 = checkBox11.checked();
                      c12 = checkBox12.checked();
                      c13 = checkBox13.checked();
                      c14 = checkBox14.checked();

                    var choices_g = [c1,c8];
                    var choices_a = [c2,c3,c4,c5,c6,c7,c9,c10,c11,c12,c13,c14];
                    var gender = ["M","F"]; 
                    var choice_gender = [];
                    var age = ["M22-","M23-26","M27-28","M29-31","M32-38","M39+","F23-","F24-26","F27-28","F29-32","F33-42","F43+"] 
                    var choice_age = [];

                    //creating an array for the selected attributes
                    choices_g.forEach(function(d,i){
                      if(d==true){
                        choice_gender.push(gender[i]);
                      }
                    });

                    choices_a.forEach(function(d,i){
                      if(d==true){
                        choice_age.push(age[i]);
                      }
                    });

                    //Deciding which output data to be used base on dropdown selection
                    if(selectValue == "ANN"){
                      devices = ANN;
                      acy = ACC[2];
                    }

                    else if(selectValue == "Logistic Regression"){
                      devices = LR;
                      acy = ACC[0];
                    }

                    else if(selectValue == "XGBoost"){
                      devices = XG;
                      acy = ACC[1];
                    }

                    //filtering data based on selection
                    newData1 = devices.filter(function(d,i){return choice_gender.includes(d.gender);});
                    newData = newData1.filter(function(d,i){return choice_age.includes(d.age);});

                    //Calling count to update the percentage values of each class  
                    checkBox1.count(newData);
                    checkBox2.count(newData);
                    checkBox3.count(newData);
                    checkBox4.count(newData);
                    checkBox5.count(newData);
                    checkBox6.count(newData);
                    checkBox7.count(newData);
                    checkBox8.count(newData);
                    checkBox9.count(newData);
                    checkBox10.count(newData);
                    checkBox11.count(newData);
                    checkBox12.count(newData);
                    checkBox13.count(newData);
                    checkBox14.count(newData);

                    //Removing the previously plotted points and text  
                    g.selectAll("circle").remove();  
                    svg.selectAll("#accuracy").remove(); 

                    //Plotting the filtered data points
                    g.selectAll("circle")
                      .data(newData).enter()
                      .append("circle")
                      .attr("cx", function (d) { return projection([d.longitude,d.latitude])[0]; })
                      .attr("cy", function (d) { return projection([d.longitude,d.latitude])[1]; })
                      .attr("r", "1px")
                      .attr("fill", function(d){if(d.gender=="M"){return "steelblue";}
                                                else if(d.gender=="F"){return "#C76D6D"}
                    });  
                    
                    //Updating the accuracy based on model  
                    svg.append('text')
                      .attr('id','accuracy')
                      .attr("x", width*0.05)
                      .attr("y", height-50)
                      .attr("text-anchor", "end")
                      .text(acy.ga)
                      .style({
                          "font-family": "Calibri",
                          "font-size":"14px",
                          "font-weight":"bold"
                      });

                    svg.append('text')
                      .attr('id','accuracy')
                      .attr("x", width*0.94)
                      .attr("y", height-50)
                      .attr("text-anchor", "end")
                      .text(acy.oa)
                      .style({
                          "font-family": "Calibri",
                          "font-size":"14px",
                          "font-weight":"bold"
                      });
                  
                };

            //Setting up each check box
            checkBox1.size(15).x(width-65).y(10).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox2.size(10).x(width-61).y(40).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox3.size(10).x(width-61).y(60).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox4.size(10).x(width-61).y(80).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox5.size(10).x(width-61).y(100).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox6.size(10).x(width-61).y(120).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox7.size(10).x(width-61).y(140).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            
            checkBox8.size(15).x(40).y(10).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox9.size(10).x(45).y(40).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox10.size(10).x(45).y(60).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox11.size(10).x(45).y(80).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox12.size(10).x(45).y(100).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox13.size(10).x(45).y(120).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            checkBox14.size(10).x(45).y(140).rx(5).ry(5).markStrokeWidth(1).boxStrokeWidth(1).checked(true).clickEvent(update);
            
            //Calling checkbox
            svg.call(checkBox1);
            svg.call(checkBox2);
            svg.call(checkBox3);
            svg.call(checkBox4);
            svg.call(checkBox5);
            svg.call(checkBox6);
            svg.call(checkBox7);
            svg.call(checkBox8);
            svg.call(checkBox9);
            svg.call(checkBox10);
            svg.call(checkBox11);
            svg.call(checkBox12);
            svg.call(checkBox13);
            svg.call(checkBox14);

            //Appending text for Labes "Gender Accuracy" and "Overall Accuracy"
            svg.append('text')
              .attr("x", width*0.1)
              .attr("y", height-80)
              .attr("text-anchor", "end")
              .text("Gender Accuracy")
              .style({
                  "font-family": "Calibri",
                  "font-size":"14px",
                  "font-weight":"bold"
              });

            svg.append('text')
              .attr("x", width*0.97)
              .attr("y", height-80)
              .attr("text-anchor", "end")
              .text("Overall Accuracy")
              .style({
                  "font-family": "Calibri",
                  "font-size":"14px",
                  "font-weight":"bold"
              });

            update();

            //Transforming and translating the map to center it
            g.attr("transform", "translate(" + projection.translate() + ")scale(" + xyz[2] + ")translate(-" + xyz[0] + ",-" + xyz[1] + ")")
                .selectAll(["#countries"])
                .style("stroke-width", 1.0 / xyz[2] + "px")
                .selectAll(".city")
                .attr("d", path.pointRadius(20.0 / xyz[2]));
    
            function get_xyz(d) {
              var bounds = path.bounds(d);
              var w_scale = (bounds[1][0] - bounds[0][0]) / width;
              var h_scale = (bounds[1][1] - bounds[0][1]) / height;
              var z = .96 / Math.max(w_scale, h_scale);
              var x = (bounds[1][0] + bounds[0][0]) / 2;
              var y = (bounds[1][1] + bounds[0][1]) / 2 + (height / z / 6);
              return [x, y, z];
            }    
            })
          })
        })
      })
    });