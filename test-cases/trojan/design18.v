module top(n0, n1, n2, n3, n4, n5, n6, n8, n9, n7, n10);
    input n0, n1, n2;
    input [13:0] n3;
    input [12:0] n4;
    input [7:0] n5, n6, n7;
    input [5:0] n8;
    input [9:0] n9;
    output [12:0] n10;
    wire n0, n1, n2;
    wire [13:0] n3;
    wire [12:0] n4;
    wire [7:0] n5, n6, n7;
    wire [5:0] n8;
    wire [9:0] n9;
    wire [12:0] n10;
    wire [3:0] n11;
    wire [3:0] n12;
    wire [7:0] n13;
    wire [12:0] n14;
    wire [3:0] n15;
    wire [7:0] n16;
    wire n17, n18, n19, n20, n21, n22, n23, n24;
    wire n25, n26, n27, n28, n29, n30, n31, n32;
    wire n33, n34, n35, n36, n37, n38, n39, n40;
    wire n41, n42, n43, n44, n45, n46, n47, n48;
    wire n49, n50, n51, n52, n53, n54, n55, n56;
    wire n57, n58, n59, n60, n61, n62, n63, n64;
    wire n65, n66, n67, n68, n69, n70, n71, n72;
    wire n73, n74, n75, n76, n77, n78, n79, n80;
    wire n81, n82, n83, n84, n85, n86, n87, n88;
    wire n89, n90, n91, n92, n93, n94, n95, n96;
    wire n97, n98, n99, n100, n101, n102, n103, n104;
    wire n105, n106, n107, n108, n109, n110, n111, n112;
    wire n113, n114, n115, n116, n117, n118, n119, n120;
    wire n121, n122, n123, n124, n125, n126, n127, n128;
    wire n129, n130, n131, n132, n133, n134, n135, n136;
    wire n137, n138, n139, n140, n141, n142, n143, n144;
    wire n145, n146, n147, n148, n149, n150, n151, n152;
    wire n153, n154, n155, n156, n157, n158, n159, n160;
    wire n161, n162, n163, n164, n165, n166, n167, n168;
    wire n169, n170, n171, n172, n173, n174, n175, n176;
    wire n177, n178, n179, n180, n181, n182, n183, n184;
    wire n185, n186, n187, n188, n189, n190, n191, n192;
    wire n193, n194, n195, n196, n197, n198, n199, n200;
    wire n201, n202, n203, n204, n205, n206, n207, n208;
    wire n209, n210, n211, n212;
    not g0(n64 ,n15[3]);
    or g1(n93 ,n66 ,n89);
    dff g2(.RN(n1), .SN(1'b1), .CK(n0), .D(n148), .Q(n10[3]));
    dff g3(.RN(n1), .SN(1'b1), .CK(n0), .D(n136), .Q(n13[3]));
    nor g4(n178 ,n4[3] ,n176);
    not g5(n174 ,n173);
    dff g6(.RN(n1), .SN(1'b1), .CK(n0), .D(n142), .Q(n12[1]));
    nor g7(n97 ,n42 ,n84);
    nor g8(n105 ,n65 ,n86);
    dff g9(.RN(n1), .SN(1'b1), .CK(n0), .D(n145), .Q(n11[0]));
    not g10(n171 ,n4[2]);
    dff g11(.RN(n1), .SN(1'b1), .CK(n0), .D(n149), .Q(n10[0]));
    nor g12(n187 ,n4[6] ,n185);
    nor g13(n14[7] ,n190 ,n191);
    nor g14(n73 ,n59 ,n58);
    or g15(n137 ,n126 ,n130);
    not g16(n34 ,n15[0]);
    not g17(n183 ,n182);
    not g18(n101 ,n100);
    nor g19(n14[6] ,n187 ,n188);
    not g20(n162 ,n4[10]);
    not g21(n59 ,n16[3]);
    not g22(n51 ,n11[2]);
    not g23(n177 ,n176);
    nor g24(n212 ,n211 ,n210);
    not g25(n186 ,n185);
    dff g26(.RN(n1), .SN(1'b1), .CK(n0), .D(n117), .Q(n15[1]));
    nor g27(n72 ,n56 ,n52);
    nor g28(n91 ,n159 ,n85);
    not g29(n160 ,n4[1]);
    xor g30(n146 ,n112 ,n14[2]);
    nor g31(n14[10] ,n199 ,n200);
    not g32(n45 ,n152);
    not g33(n55 ,n13[2]);
    not g34(n46 ,n158);
    or g35(n24 ,n22 ,n23);
    nor g36(n190 ,n4[7] ,n188);
    nor g37(n31 ,n12[2] ,n29);
    nor g38(n197 ,n163 ,n195);
    nor g39(n80 ,n11[1] ,n11[3]);
    nor g40(n116 ,n98 ,n91);
    dff g41(.RN(n1), .SN(1'b1), .CK(n0), .D(n139), .Q(n12[0]));
    xor g42(n150 ,n113 ,n14[1]);
    not g43(n180 ,n179);
    xnor g44(n122 ,n83 ,n15[0]);
    not g45(n21 ,n15[2]);
    nor g46(n14[5] ,n184 ,n185);
    nor g47(n203 ,n169 ,n201);
    nor g48(n114 ,n72 ,n95);
    nor g49(n133 ,n44 ,n110);
    or g50(n208 ,n204 ,n205);
    not g51(n61 ,n12[2]);
    dff g52(.RN(n1), .SN(1'b1), .CK(n0), .D(n5[3]), .Q(n16[3]));
    not g53(n83 ,n84);
    nor g54(n209 ,n208 ,n206);
    dff g55(.RN(n1), .SN(1'b1), .CK(n0), .D(n118), .Q(n15[2]));
    not g56(n22 ,n15[3]);
    or g57(n121 ,n96 ,n106);
    nor g58(n145 ,n90 ,n116);
    nor g59(n199 ,n4[10] ,n197);
    not g60(n33 ,n15[1]);
    not g61(n50 ,n11[3]);
    nor g62(n206 ,n3[11] ,n3[10]);
    xor g63(n154 ,n12[3] ,n32);
    dff g64(.RN(n1), .SN(1'b1), .CK(n0), .D(n137), .Q(n13[2]));
    not g65(n17 ,n12[2]);
    not g66(n49 ,n154);
    not g67(n30 ,n29);
    not g68(n163 ,n4[9]);
    not g69(n63 ,n15[1]);
    nor g70(n108 ,n46 ,n83);
    not g71(n170 ,n4[3]);
    or g72(n117 ,n109 ,n108);
    nor g73(n103 ,n61 ,n86);
    not g74(n164 ,n4[5]);
    nor g75(n86 ,n11[1] ,n79);
    dff g76(.RN(n1), .SN(1'b1), .CK(n0), .D(n144), .Q(n11[3]));
    not g77(n81 ,n80);
    or g78(n89 ,n78 ,n70);
    nor g79(n147 ,n81 ,n145);
    xor g80(n149 ,n4[0] ,n114);
    nor g81(n185 ,n164 ,n183);
    nor g82(n128 ,n52 ,n111);
    nor g83(n157 ,n39 ,n40);
    nor g84(n124 ,n50 ,n98);
    nor g85(n112 ,n74 ,n93);
    dff g86(.RN(n1), .SN(1'b1), .CK(n0), .D(n122), .Q(n15[0]));
    nor g87(n182 ,n165 ,n180);
    nor g88(n135 ,n55 ,n110);
    nor g89(n67 ,n16[3] ,n13[3]);
    nor g90(n120 ,n80 ,n99);
    or g91(n20 ,n18 ,n19);
    nor g92(n194 ,n167 ,n192);
    nor g93(n90 ,n155 ,n83);
    nor g94(n36 ,n15[1] ,n15[0]);
    nor g95(n207 ,n3[13] ,n3[12]);
    not g96(n78 ,n77);
    not g97(n85 ,n86);
    not g98(n189 ,n188);
    nor g99(n66 ,n16[2] ,n13[2]);
    nor g100(n152 ,n29 ,n28);
    not g101(n169 ,n4[11]);
    nor g102(n68 ,n11[1] ,n11[2]);
    nor g103(n113 ,n75 ,n94);
    not g104(n42 ,n15[2]);
    nor g105(n14[4] ,n181 ,n182);
    or g106(n141 ,n103 ,n133);
    nor g107(n75 ,n43 ,n53);
    nor g108(n76 ,n16[0] ,n13[0]);
    or g109(n143 ,n128 ,n132);
    nor g110(n14[8] ,n193 ,n194);
    nor g111(n131 ,n52 ,n110);
    nor g112(n127 ,n53 ,n111);
    not g113(n52 ,n13[0]);
    dff g114(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[4]), .Q(n10[4]));
    nor g115(n158 ,n37 ,n36);
    or g116(n118 ,n97 ,n107);
    nor g117(n123 ,n12[0] ,n110);
    not g118(n19 ,n12[1]);
    dff g119(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[7]), .Q(n10[7]));
    nor g120(n100 ,n11[3] ,n88);
    nor g121(n200 ,n162 ,n198);
    not g122(n27 ,n12[2]);
    not g123(n166 ,n4[6]);
    dff g124(.RN(n1), .SN(1'b1), .CK(n0), .D(n138), .Q(n13[1]));
    nor g125(n144 ,n100 ,n124);
    not g126(n165 ,n4[4]);
    nor g127(n196 ,n4[9] ,n194);
    nor g128(n102 ,n60 ,n86);
    nor g129(n107 ,n47 ,n83);
    not g130(n65 ,n12[3]);
    nor g131(n134 ,n45 ,n110);
    not g132(n205 ,n3[12]);
    nor g133(n125 ,n58 ,n111);
    dff g134(.RN(n1), .SN(1'b1), .CK(n0), .D(n146), .Q(n10[2]));
    not g135(n25 ,n12[1]);
    not g136(n18 ,n12[3]);
    nor g137(n82 ,n11[2] ,n77);
    dff g138(.RN(n1), .SN(1'b1), .CK(n0), .D(n150), .Q(n10[1]));
    nor g139(n37 ,n33 ,n34);
    nor g140(n14[3] ,n178 ,n179);
    not g141(n62 ,n16[2]);
    not g142(n60 ,n12[1]);
    not g143(n211 ,n2);
    not g144(n168 ,n212);
    nor g145(n87 ,n50 ,n68);
    or g146(n70 ,n50 ,n11[2]);
    dff g147(.RN(n1), .SN(1'b1), .CK(n0), .D(n5[0]), .Q(n16[0]));
    nor g148(n115 ,n73 ,n92);
    nor g149(n153 ,n31 ,n32);
    dff g150(.RN(n1), .SN(1'b1), .CK(n0), .D(n119), .Q(n11[2]));
    or g151(n140 ,n105 ,n129);
    dff g152(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[12]), .Q(n10[12]));
    not g153(n54 ,n11[0]);
    not g154(n195 ,n194);
    nor g155(n202 ,n4[11] ,n200);
    nor g156(n14[1] ,n173 ,n172);
    dff g157(.RN(n1), .SN(1'b1), .CK(n0), .D(n141), .Q(n12[2]));
    not g158(n99 ,n98);
    not g159(n48 ,n156);
    nor g160(n188 ,n166 ,n186);
    or g161(n71 ,n41 ,n11[0]);
    xor g162(n14[12] ,n4[12] ,n203);
    nor g163(n193 ,n4[8] ,n191);
    or g164(n94 ,n69 ,n89);
    not g165(n167 ,n4[8]);
    dff g166(.RN(n1), .SN(1'b1), .CK(n0), .D(n140), .Q(n12[3]));
    dff g167(.RN(n1), .SN(1'b1), .CK(n0), .D(n151), .Q(n11[1]));
    nor g168(n155 ,n21 ,n24);
    not g169(n161 ,n4[7]);
    nor g170(n173 ,n160 ,n168);
    nor g171(n184 ,n4[5] ,n182);
    or g172(n95 ,n76 ,n89);
    or g173(n210 ,n207 ,n209);
    nor g174(n130 ,n53 ,n110);
    or g175(n151 ,n120 ,n147);
    nor g176(n98 ,n11[0] ,n87);
    nor g177(n191 ,n161 ,n189);
    nor g178(n32 ,n27 ,n30);
    nor g179(n39 ,n15[2] ,n37);
    nor g180(n126 ,n55 ,n111);
    or g181(n92 ,n67 ,n89);
    nor g182(n29 ,n25 ,n26);
    nor g183(n14[11] ,n202 ,n203);
    not g184(n47 ,n157);
    nor g185(n175 ,n4[2] ,n173);
    not g186(n44 ,n153);
    not g187(n198 ,n197);
    nor g188(n106 ,n48 ,n83);
    dff g189(.RN(n1), .SN(1'b1), .CK(n0), .D(n5[2]), .Q(n16[2]));
    nor g190(n74 ,n62 ,n55);
    nor g191(n129 ,n49 ,n110);
    nor g192(n176 ,n171 ,n174);
    or g193(n138 ,n127 ,n131);
    nor g194(n14[2] ,n175 ,n176);
    nor g195(n119 ,n82 ,n101);
    nor g196(n111 ,n54 ,n85);
    dff g197(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[9]), .Q(n10[9]));
    not g198(n192 ,n191);
    nor g199(n179 ,n170 ,n177);
    dff g200(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[6]), .Q(n10[6]));
    not g201(n35 ,n15[2]);
    nor g202(n104 ,n57 ,n86);
    nor g203(n109 ,n63 ,n84);
    nor g204(n77 ,n54 ,n41);
    not g205(n38 ,n37);
    not g206(n43 ,n16[1]);
    not g207(n26 ,n12[0]);
    dff g208(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[10]), .Q(n10[10]));
    not g209(n58 ,n13[3]);
    not g210(n110 ,n111);
    dff g211(.RN(n1), .SN(1'b1), .CK(n0), .D(n143), .Q(n13[0]));
    or g212(n142 ,n102 ,n134);
    dff g213(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[8]), .Q(n10[8]));
    not g214(n204 ,n3[13]);
    nor g215(n96 ,n64 ,n84);
    dff g216(.RN(n1), .SN(1'b1), .CK(n0), .D(n5[1]), .Q(n16[1]));
    nor g217(n69 ,n16[1] ,n13[1]);
    nor g218(n14[9] ,n196 ,n197);
    nor g219(n181 ,n4[4] ,n179);
    xor g220(n148 ,n115 ,n14[3]);
    nor g221(n28 ,n12[1] ,n12[0]);
    not g222(n41 ,n11[1]);
    dff g223(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[11]), .Q(n10[11]));
    nor g224(n40 ,n35 ,n38);
    or g225(n139 ,n104 ,n123);
    or g226(n79 ,n11[2] ,n11[3]);
    dff g227(.RN(n1), .SN(1'b1), .CK(n0), .D(n121), .Q(n15[3]));
    nor g228(n132 ,n56 ,n110);
    xor g229(n156 ,n15[3] ,n40);
    not g230(n201 ,n200);
    not g231(n57 ,n12[0]);
    nor g232(n159 ,n17 ,n20);
    nor g233(n88 ,n51 ,n78);
    not g234(n53 ,n13[1]);
    or g235(n136 ,n125 ,n135);
    dff g236(.RN(n1), .SN(1'b1), .CK(n0), .D(n14[5]), .Q(n10[5]));
    nor g237(n172 ,n4[1] ,n212);
    not g238(n56 ,n16[0]);
    nor g239(n84 ,n71 ,n79);
    not g240(n23 ,n15[1]);
endmodule
