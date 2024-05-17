//
// Created by ubuntu on 24-5-17.
//
#include "libtrt.hpp"
#include "progressbar.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/extension.h"
#include <iostream>

#define SET_ATTR_TO(m, name, value) m.attr(#name) = value

namespace py = pybind11;

// clang-format off
constexpr float alphas_cumprod[1000] = {0.9991499781608582, 0.9982922673225403, 0.9974268674850464, 0.9965537190437317, 0.9956728219985962, 0.9947841167449951, 0.9938876032829285, 0.9929832816123962, 0.9920711517333984, 0.9911511540412903, 0.9902232885360718, 0.9892874956130981, 0.9883438348770142, 0.987392246723175, 0.9864327311515808, 0.9854652285575867, 0.9844897389411926, 0.9835063219070435, 0.9825148582458496, 0.9815154075622559, 0.9805079102516174, 0.9794923663139343, 0.9784687161445618, 0.9774370193481445, 0.9763972759246826, 0.9753494262695312, 0.9742934703826904, 0.9732293486595154, 0.9721570611000061, 0.9710767269134521, 0.9699881672859192, 0.9688915014266968, 0.9677866101264954, 0.9666735529899597, 0.9655522704124451, 0.9644228219985962, 0.9632851481437683, 0.9621393084526062, 0.9609851837158203, 0.9598228335380554, 0.9586522579193115, 0.9574733972549438, 0.9562862515449524, 0.9550909399986267, 0.9538873434066772, 0.9526754021644592, 0.9514552354812622, 0.9502268433570862, 0.9489901661872864, 0.947745144367218, 0.9464918971061707, 0.9452303051948547, 0.9439604878425598, 0.9426823258399963, 0.9413958787918091, 0.940101146697998, 0.9387981295585632, 0.9374868273735046, 0.9361672401428223, 0.9348393082618713, 0.9335030913352966, 0.9321586489677429, 0.9308059215545654, 0.9294448494911194, 0.9280755519866943, 0.9266979694366455, 0.9253121614456177, 0.9239179491996765, 0.9225155711174011, 0.921104907989502, 0.9196860194206238, 0.9182589054107666, 0.9168235063552856, 0.9153799414634705, 0.9139280915260315, 0.9124680757522583, 0.9109998345375061, 0.9095233678817749, 0.9080387353897095, 0.906545877456665, 0.9050449132919312, 0.9035358428955078, 0.9020185470581055, 0.9004931449890137, 0.8989596366882324, 0.8974180221557617, 0.8958683013916016, 0.8943105340003967, 0.8927447199821472, 0.8911707997322083, 0.8895888328552246, 0.8879988789558411, 0.8864009380340576, 0.8847950100898743, 0.8831810355186462, 0.8815591335296631, 0.8799293637275696, 0.8782916069030762, 0.8766459822654724, 0.8749924898147583, 0.8733311295509338, 0.8716619610786438, 0.8699849247932434, 0.8683001399040222, 0.8666075468063354, 0.8649072051048279, 0.8631991147994995, 0.8614833354949951, 0.8597598671913147, 0.8580287098884583, 0.8562899827957153, 0.8545436263084412, 0.8527897000312805, 0.8510282039642334, 0.8492591381072998, 0.8474825620651245, 0.8456985354423523, 0.8439070582389832, 0.8421081304550171, 0.8403018116950989, 0.8384881615638733, 0.8366671800613403, 0.8348388671875, 0.8330032825469971, 0.8311604261398315, 0.8293103575706482, 0.8274531364440918, 0.8255887627601624, 0.8237172365188599, 0.8218386769294739, 0.8199530243873596, 0.8180603981018066, 0.8161607384681702, 0.8142541646957397, 0.8123406767845154, 0.8104203343391418, 0.8084931373596191, 0.8065590858459473, 0.8046183586120605, 0.8026708364486694, 0.8007166385650635, 0.7987557649612427, 0.7967883348464966, 0.7948142290115356, 0.792833685874939, 0.790846586227417, 0.7888530492782593, 0.7868530750274658, 0.7848467230796814, 0.7828341126441956, 0.7808151245117188, 0.7787899374961853, 0.77675861120224, 0.7747210264205933, 0.7726773619651794, 0.7706276774406433, 0.7685719132423401, 0.7665101885795593, 0.764442503452301, 0.7623689770698547, 0.7602895498275757, 0.7582043409347534, 0.7561134099960327, 0.7540167570114136, 0.7519145011901855, 0.7498065829277039, 0.7476931214332581, 0.7455741763114929, 0.7434498071670532, 0.7413199543952942, 0.7391847968101501, 0.7370443344116211, 0.7348986268043518, 0.7327476739883423, 0.7305915951728821, 0.7284303903579712, 0.7262641787528992, 0.724092960357666, 0.7219167351722717, 0.7197356820106506, 0.717549741268158, 0.715359091758728, 0.7131636738777161, 0.7109635472297668, 0.7087588906288147, 0.7065496444702148, 0.7043358683586121, 0.7021176815032959, 0.6998950839042664, 0.697668194770813, 0.695436954498291, 0.6932015419006348, 0.6909619569778442, 0.6887182593345642, 0.6864705085754395, 0.6842187643051147, 0.6819631457328796, 0.6797035932540894, 0.6774402260780334, 0.6751731634140015, 0.6729023456573486, 0.6706279516220093, 0.6683499813079834, 0.6660684943199158, 0.6637835502624512, 0.6614952087402344, 0.6592035293579102, 0.6569086313247681, 0.6546105146408081, 0.6523092985153198, 0.6500049233436584, 0.6476975679397583, 0.6453872323036194, 0.6430740356445312, 0.6407580375671387, 0.6384392380714417, 0.6361177563667297, 0.6337936520576477, 0.6314669251441956, 0.6291377544403076, 0.6268061399459839, 0.6244720816612244, 0.6221357583999634, 0.6197972297668457, 0.6174564957618713, 0.6151136159896851, 0.6127687692642212, 0.6104218363761902, 0.6080729961395264, 0.6057223677635193, 0.6033698916435242, 0.6010156869888306, 0.598659873008728, 0.5963024497032166, 0.5939435362815857, 0.5915831327438354, 0.5892213582992554, 0.5868582725524902, 0.58449387550354, 0.5821283459663391, 0.5797616839408875, 0.5773940086364746, 0.5750252604484558, 0.5726556777954102, 0.5702852010726929, 0.5679139494895935, 0.5655420422554016, 0.5631694197654724, 0.5607962608337402, 0.5584225654602051, 0.5560484528541565, 0.5536739230155945, 0.5512991547584534, 0.5489240884780884, 0.5465489029884338, 0.5441735982894897, 0.5417982339859009, 0.5394229292869568, 0.5370477437973022, 0.5346726775169373, 0.5322978496551514, 0.5299233794212341, 0.5275492668151855, 0.5251755714416504, 0.5228024125099182, 0.5204298496246338, 0.5180578827857971, 0.5156866312026978, 0.5133161544799805, 0.5109465718269348, 0.5085778832435608, 0.506210207939148, 0.5038435459136963, 0.5014780163764954, 0.49911367893218994, 0.4967505931854248, 0.4943888187408447, 0.49202844500541687, 0.4896695017814636, 0.48731210827827454, 0.484956294298172, 0.48260214924812317, 0.48024967312812805, 0.4778990149497986, 0.47555020451545715, 0.4732033312320709, 0.4708584249019623, 0.4685156047344208, 0.4661748707294464, 0.46383631229400635, 0.46150001883506775, 0.4591660499572754, 0.45683443546295166, 0.4545052647590637, 0.45217859745025635, 0.4498545229434967, 0.4475330710411072, 0.44521430134773254, 0.44289830327033997, 0.4405851364135742, 0.4382748603820801, 0.4359675347805023, 0.4336632192134857, 0.43136200308799744, 0.4290638864040375, 0.4267689883708954, 0.42447736859321594, 0.42218905687332153, 0.4199041426181793, 0.41762271523475647, 0.415344774723053, 0.4130703806877136, 0.41079962253570557, 0.4085325598716736, 0.40626925230026245, 0.40400975942611694, 0.40175411105155945, 0.3995024263858795, 0.39725470542907715, 0.3950110077857971, 0.392771452665329, 0.39053604006767273, 0.38830482959747314, 0.3860779106616974, 0.38385531306266785, 0.3816371262073517, 0.3794233798980713, 0.37721413373947144, 0.3750094175338745, 0.3728093206882477, 0.3706139028072357, 0.368423193693161, 0.36623725295066833, 0.36405614018440247, 0.3618799149990082, 0.35970863699913025, 0.3575423061847687, 0.355381041765213, 0.35322487354278564, 0.35107383131980896, 0.34892794489860535, 0.34678736329078674, 0.34465205669403076, 0.3425220549106598, 0.3403974771499634, 0.33827832341194153, 0.336164653301239, 0.3340565264225006, 0.3319540023803711, 0.32985708117485046, 0.32776588201522827, 0.32568037509918213, 0.3236006200313568, 0.3215267062187195, 0.31945866346359253, 0.31739652156829834, 0.3153403103351593, 0.3132901191711426, 0.31124594807624817, 0.30920785665512085, 0.3071759343147278, 0.3051501214504242, 0.30313053727149963, 0.3011172115802765, 0.29911017417907715, 0.297109454870224, 0.2951151132583618, 0.2931271493434906, 0.2911456525325775, 0.28917062282562256, 0.2872021496295929, 0.28524020314216614, 0.28328487277030945, 0.2813361883163452, 0.2793941795825958, 0.2774588465690613, 0.27553027868270874, 0.2736084759235382, 0.27169346809387207, 0.2697853147983551, 0.2678840458393097, 0.26598966121673584, 0.2641022503376007, 0.2622217833995819, 0.2603483200073242, 0.25848188996315, 0.2566225230693817, 0.25477027893066406, 0.2529251277446747, 0.25108712911605835, 0.24925631284713745, 0.24743269383907318, 0.24561631679534912, 0.24380719661712646, 0.2420053631067276, 0.24021084606647491, 0.2384236603975296, 0.23664385080337524, 0.23487140238285065, 0.2331063598394394, 0.23134878277778625, 0.22959862649440765, 0.22785596549510956, 0.22612079977989197, 0.22439312934875488, 0.22267301380634308, 0.22096046805381775, 0.2192554920911789, 0.2175581157207489, 0.21586833894252777, 0.21418620645999908, 0.21251173317432404, 0.21084491908550262, 0.20918579399585724, 0.20753435790538788, 0.20589064061641693, 0.2042546570301056, 0.20262642204761505, 0.2010059505701065, 0.19939322769641876, 0.1977883130311966, 0.1961911916732788, 0.1946018636226654, 0.19302035868167877, 0.1914467066526413, 0.1898808628320694, 0.18832288682460785, 0.18677276372909546, 0.18523052334785461, 0.18369615077972412, 0.18216967582702637, 0.18065106868743896, 0.1791403740644455, 0.17763757705688477, 0.17614269256591797, 0.1746557354927063, 0.17317669093608856, 0.17170557379722595, 0.17024238407611847, 0.1687871217727661, 0.16733980178833008, 0.16590042412281036, 0.16446897387504578, 0.1630454808473587, 0.16162993013858795, 0.16022230684757233, 0.15882264077663422, 0.15743091702461243, 0.15604713559150696, 0.15467128157615662, 0.1533033847808838, 0.1519434154033661, 0.1505913883447647, 0.14924727380275726, 0.14791110157966614, 0.14658284187316895, 0.14526250958442688, 0.14395008981227875, 0.14264558255672455, 0.1413489729166031, 0.14006026089191437, 0.1387794315814972, 0.13750649988651276, 0.13624143600463867, 0.13498423993587494, 0.13373489677906036, 0.13249340653419495, 0.1312597692012787, 0.130033940076828, 0.12881594896316528, 0.12760576605796814, 0.12640337646007538, 0.125208780169487, 0.12402196228504181, 0.12284290045499802, 0.12167160212993622, 0.12050803750753403, 0.11935220658779144, 0.11820407956838608, 0.11706364899873734, 0.11593090742826462, 0.11480583995580673, 0.11368842422962189, 0.11257864534854889, 0.11147648841142654, 0.11038193851709366, 0.10929498076438904, 0.10821560025215149, 0.10714376717805862, 0.10607948154211044, 0.10502271354198456, 0.10397344082593918, 0.10293165594339371, 0.10189734399318695, 0.10087046772241592, 0.09985101968050003, 0.09883897751569748, 0.09783432632684708, 0.09683703631162643, 0.09584709256887436, 0.09486446529626846, 0.09388914704322815, 0.09292110055685043, 0.09196032583713531, 0.09100677818059921, 0.09006044268608093, 0.08912129700183868, 0.08818932622671127, 0.08726449310779572, 0.08634677529335022, 0.08543615788221359, 0.08453261107206345, 0.08363611996173859, 0.08274663984775543, 0.08186416327953339, 0.08098865300416946, 0.08012009412050247, 0.07925844937562943, 0.07840370386838913, 0.0775558203458786, 0.07671478390693665, 0.07588056474924088, 0.0750531330704689, 0.07423245161771774, 0.0734185129404068, 0.07261127978563309, 0.07181072235107422, 0.0710168182849884, 0.07022953033447266, 0.06944884359836578, 0.0686747208237648, 0.06790713220834732, 0.06714605540037155, 0.06639145314693451, 0.0656433030962944, 0.06490156799554825, 0.06416622549295425, 0.06343724578619003, 0.0627145916223526, 0.06199824437499046, 0.06128816679120064, 0.06058432161808014, 0.05988669395446777, 0.05919523909687996, 0.058509938418865204, 0.057830750942230225, 0.057157646864652634, 0.05649060010910034, 0.05582957714796066, 0.05517454817891121, 0.054525475949048996, 0.05388233810663223, 0.053245093673467636, 0.05261371284723282, 0.05198816955089569, 0.05136842280626297, 0.050754446536302567, 0.05014621093869209, 0.049543678760528564, 0.048946816474199295, 0.048355598002672195, 0.04776998609304428, 0.047189947217702866, 0.04661545157432556, 0.04604646563529968, 0.04548295959830284, 0.04492489621043205, 0.04437224566936493, 0.043824970722198486, 0.043283041566610336, 0.04274642467498779, 0.04221509024500847, 0.04168900102376938, 0.04116811975836754, 0.04065242409706116, 0.04014187306165695, 0.039636436849832535, 0.03913607820868492, 0.03864077106118202, 0.038150474429130554, 0.03766516223549843, 0.03718479350209236, 0.036709342151880264, 0.03623877465724945, 0.03577305004000664, 0.035312145948410034, 0.03485602140426636, 0.03440464287996292, 0.03395798057317734, 0.03351600095629692, 0.033078670501708984, 0.03264595940709114, 0.03221782669425011, 0.03179424628615379, 0.031375180929899216, 0.030960602685809135, 0.030550474300980568, 0.030144764110445976, 0.029743440449237823, 0.029346467927098274, 0.02895381674170494, 0.028565451502799988, 0.02818134054541588, 0.027801452204585075, 0.027425752952694893, 0.027054211124777794, 0.026686793193221092, 0.02632346749305725, 0.025964202359318733, 0.025608966127038002, 0.02525772526860237, 0.024910448119044304, 0.024567103013396263, 0.024227658286690712, 0.023892080411314964, 0.023560339584946632, 0.02323240600526333, 0.022908244282007217, 0.02258782461285591, 0.022271115332841873, 0.021958086639642715, 0.0216487068682909, 0.021342944353818893, 0.021040767431259155, 0.020742148160934448, 0.020447051152586937, 0.02015545219182968, 0.019867315888404846, 0.019582612439990044, 0.019301312044262886, 0.019023386761546135, 0.018748803064227104, 0.018477533012628555, 0.0182095468044281, 0.017944814637303352, 0.017683304846286774, 0.017424993216991425, 0.01716984622180462, 0.01691783405840397, 0.016668930649757385, 0.01642310619354248, 0.016180330887436867, 0.015940576791763306, 0.01570381596684456, 0.015470017679035664, 0.01523915771394968, 0.015011204406619072, 0.01478613168001175, 0.0145639106631279, 0.014344514347612858, 0.014127915725111961, 0.013914086855947971, 0.013702998869121075, 0.013494627550244331, 0.013288944028317928, 0.013085922226309776, 0.012885536067187786, 0.01268775761127472, 0.012492561712861061, 0.012299920432269573, 0.012109809555113316, 0.011922203004360199, 0.011737074702978134, 0.011554399505257607, 0.011374151334166527, 0.01119630504399538, 0.011020835489034653, 0.010847717523574829, 0.01067692693322897, 0.01050843857228756, 0.010342227295041084, 0.010178270749747753, 0.010016543790698051, 0.009857021272182465, 0.009699681773781776, 0.009544499218463898, 0.009391451254487038, 0.009240513667464256, 0.009091665036976337, 0.008944880217313766, 0.008800136856734753, 0.00865741353482008, 0.008516686968505383, 0.008377933874726295, 0.008241133764386177, 0.008106262423098087, 0.007973299361765385, 0.007842223159968853, 0.007713011000305414, 0.007585642393678427, 0.0074600959196686745, 0.007336350157856941, 0.0072143846191465855, 0.007094177883118391, 0.006975709926337004, 0.006858960725367069, 0.0067439088597893715, 0.0066305347718298435, 0.006518818903714418, 0.006408741232007742, 0.00630028173327446, 0.0061934213154017925, 0.00608814088627696, 0.0059844208881258965, 0.0058822426944971085, 0.005781587213277817, 0.005682435818016529, 0.005584769882261753, 0.005488571710884571, 0.005393822677433491, 0.005300504621118307, 0.0052085998468101025, 0.005118090659379959, 0.005028959829360247, 0.004941189661622047, 0.004854762926697731, 0.004769662395119667, 0.0046858717687428, 0.004603373818099499, 0.004522151779383421, 0.004442189820110798, 0.0043634711764752865, 0.004285980015993118, 0.0042097000405192375, 0.004134615883231163, 0.004060711245983839, 0.003987971227616072, 0.0039163799956440926, 0.0038459226489067078, 0.00377658405341208, 0.0037083495408296585, 0.0036412037443369627, 0.003575132228434086, 0.003510120790451765, 0.00344615476205945, 0.0033832204062491655, 0.0033213035203516483, 0.0032603899016976357, 0.0032004662789404392, 0.00314151868224144, 0.003083533840253949, 0.0030264982488006353, 0.0029703988693654537, 0.0029152228962630033, 0.002860957058146596, 0.0028075887821614742, 0.002755105495452881, 0.002703494392335415, 0.0026527433656156063, 0.002602840308099985, 0.0025537731125950813, 0.0025055294390767813, 0.002458097878843546, 0.0024114667903631926, 0.002365624299272895, 0.0023205592297017574, 0.0022762601729482412, 0.0022327161859720945, 0.002189916092902422, 0.0021478489506989717, 0.002106503816321492, 0.002065870212391019, 0.002025937894359231, 0.001986695919185877, 0.0019481343915686011, 0.001910243066959083, 0.0018730118172243237, 0.0018364308634772897, 0.0018004901940003037, 0.0017651801463216543, 0.0017304914072155952, 0.001696414197795093, 0.0016629394376650453, 0.0016300578135997057, 0.0015977602452039719, 0.001566037768498063, 0.0015348814195021987, 0.0015042824670672417, 0.001474232180044055, 0.001444722176529467, 0.0014157438417896628, 0.0013872889103367925, 0.0013593491166830063, 0.0013319164281710982, 0.0013049828121438622, 0.0012785402359440923, 0.001252580899745226, 0.0012270971201360226, 0.0012020813301205635, 0.0011775260791182518, 0.0011534236837178469, 0.001129767159000039, 0.0011065491707995534, 0.0010837623849511147, 0.0010614001657813787, 0.0010394552955403924, 0.0010179209057241678, 0.0009967905934900045, 0.000976057315710932, 0.0009557147277519107, 0.0009357563103549182, 0.0009161756606772542, 0.0008969665504992008, 0.0008781226351857185, 0.0008596379193477333, 0.00084150634938851, 0.0008237219299189746, 0.0008062788401730359, 0.0007891712011769414, 0.0007723934249952435, 0.0007559398072771728, 0.0007398048765026033, 0.0007239831611514091, 0.0007084693061187863, 0.0006932578980922699, 0.0006783438147976995, 0.0006637218757532537, 0.0006493869586847723, 0.0006353341159410775, 0.0006215584580786526, 0.000608055095653981, 0.0005948191974312067, 0.0005818460485897958, 0.0005691311089321971, 0.0005566697218455374, 0.0005444574635475874, 0.0005324897938407958, 0.0005207624053582549, 0.0005092710489407182, 0.000498011417221278, 0.0004869793192483485, 0.0004761707386933267, 0.0004655816010199487, 0.0004552078898996115, 0.00044504570541903377, 0.0004350912058725953, 0.00042534060776233673, 0.0004157901566941291, 0.000406436127377674, 0.00039727496914565563, 0.0003883031022269279, 0.00037951700505800545, 0.0003709132724907249, 0.00036248844116926193, 0.0003542392223607749, 0.0003461622982285917, 0.0003382544673513621, 0.0003305125283077359, 0.0003229333378840238, 0.0003155138692818582, 0.0003082510665990412, 0.00030114196124486625, 0.0002941835846286267, 0.0002873731136787683, 0.0002807076962199062, 0.0002741845091804862, 0.00026780087500810623, 0.0002615540870465338, 0.00025544146774336696, 0.000249460426857695, 0.0002436084469081834, 0.000237882966757752, 0.00023228151258081198, 0.00022680169786326587, 0.00022144109243527055, 0.0002161973825423047, 0.00021106822532601655, 0.00020605137979146093, 0.0002011446194956079, 0.000196345747099258, 0.0001916526089189574, 0.00018706310947891325, 0.00018257515330333263, 0.0001781867176759988, 0.00017389579443261027, 0.00016970040451269597, 0.00016559864161536098, 0.00016158858488779515, 0.0001576683862367645, 0.000153836197569035, 0.00015009022899903357, 0.00014642873429693282, 0.0001428499526809901, 0.00013935219612903893, 0.00013593379117082804, 0.00013259310799185187, 0.00012932851677760482, 0.00012613844592124224, 0.00012302136747166514, 0.00011997571709798649, 0.00011700002505676821, 0.00011409282888052985, 0.00011125267337774858, 0.00010847815428860486, 0.0001057678964571096, 0.00010312051745131612, 0.00010053470759885386, 9.800913539947942e-05, 9.554252756061032e-05, 9.313362534157932e-05, 9.078119182959199e-05, 8.848401193972677e-05, 8.624090696685016e-05, 8.405069820582867e-05, 8.191225060727447e-05, 7.9824443673715e-05, 7.778617145959288e-05, 7.579635712318122e-05, 7.385395292658359e-05, 7.195791113190353e-05, 7.010721310507506e-05, 6.830087659182027e-05, 6.653791933786124e-05, 6.48173809167929e-05, 6.313832273008302e-05, 6.149982800707221e-05, 5.990099816699512e-05, 5.834094190504402e-05, 5.6818800658220425e-05, 5.533372313948348e-05, 5.388488352764398e-05, 5.247146691544913e-05, 5.109266930958256e-05, 4.974771582055837e-05, 4.843583519686945e-05, 4.7156281652860343e-05, 4.5908320316812024e-05, 4.469122723094188e-05, 4.3504300265340135e-05, 4.2346851842012256e-05, 4.121820893487893e-05, 4.011770215583965e-05, 3.904469122062437e-05, 3.799853220698424e-05, 3.6978613934479654e-05, 3.5984321584692225e-05, 3.501505852909759e-05, 3.407024632906541e-05, 3.314931382192299e-05, 3.225170439691283e-05, 3.137686871923506e-05, 3.052426836802624e-05, 2.9693388569285162e-05, 2.888370909204241e-05, 2.8094733352190815e-05, 2.732596840360202e-05, 2.6576934033073485e-05, 2.5847164579317905e-05, 2.5136196200037375e-05, 2.4443583242828026e-05, 2.3768881874275394e-05, 2.3111664631869644e-05, 2.2471511329058558e-05, 2.1848010874236934e-05, 2.1240759451757185e-05, 2.0649364159908146e-05, 2.007344301091507e-05, 1.951261583599262e-05, 1.8966515199281275e-05, 1.8434784578857943e-05, 1.7917071090778336e-05, 1.741302912705578e-05, 1.692232399364002e-05, 1.6444628272438422e-05, 1.5979620002326556e-05, 1.5526986317127012e-05, 1.508641889813589e-05, 1.4657619431091007e-05, 1.4240294149203692e-05, 1.3834158380632289e-05, 1.343893109151395e-05, 1.3054340342932846e-05, 1.2680117833951954e-05, 1.2316003449086566e-05, 1.1961742529820185e-05, 1.1617086784099229e-05, 1.1281793376838323e-05, 1.0955623110930901e-05, 1.0638345884217415e-05, 1.0329735232517123e-05, 1.0029569239122793e-05, 9.737632353790104e-06, 9.453713573748246e-06, 9.177607353194617e-06, 8.909112693800125e-06, 8.648033144709188e-06, 8.394177712034434e-06, 8.147359039867297e-06, 7.90739431977272e-06, 7.674106200283859e-06, 7.447320058417972e-06, 7.2268662734131794e-06, 7.01258022672846e-06, 6.804299118812196e-06, 6.601866061828332e-06, 6.4051268964249175e-06, 6.21393155597616e-06, 6.028133611835074e-06, 5.8475907280808315e-06, 5.672163297276711e-06};
// clang-format on

inline at::TensorOptions get_options(const at::ScalarType& dtype, const at::DeviceType& device)
{
    return at::TensorOptions().dtype(dtype).device(device);
}

class TextEmbeddingModel {
public:
    int batch_size;
    int seq_len;
    int emb_dim;

private:
    libtrt::Engine m_engine;

    at::Tensor m_input_ids;
    at::Tensor m_attn_mask;
    at::Tensor m_text_emb;

    std::vector<int> m_negative_ids_vec;

public:
    TextEmbeddingModel(const std::string& engine_path): m_engine(engine_path, 0, 0)
    {
        batch_size           = m_engine.GetInput(0)->GetDims(0);
        seq_len              = m_engine.GetInput(0)->GetDims(1);
        emb_dim              = m_engine.GetOutput(0)->GetDims(2);
        int*   input_ids_ptr = reinterpret_cast<int*>(m_engine.GetInput(0)->GetPtr());
        float* attn_mask_ptr = reinterpret_cast<float*>(m_engine.GetInput(1)->GetPtr());
        float* text_emb_ptr  = reinterpret_cast<float*>(m_engine.GetOutput(0)->GetPtr());

        m_input_ids = torch::from_blob(input_ids_ptr, {batch_size, seq_len}, get_options(torch::kInt32, torch::kCUDA));
        m_attn_mask = torch::from_blob(attn_mask_ptr, {batch_size, seq_len}, get_options(torch::kBool, torch::kCUDA));
        m_text_emb =
            torch::from_blob(text_emb_ptr, {batch_size, seq_len, emb_dim}, get_options(torch::kFloat32, torch::kCUDA));
    }

    void set_default_negative_ids(const std::vector<int>& negative_ids)
    {
        m_negative_ids_vec = negative_ids;
        if (m_negative_ids_vec.size() < seq_len) {
            m_negative_ids_vec.resize(seq_len, 0);
        }
    }

    at::Tensor forward(const std::vector<int>& positive_ids, const std::optional<std::vector<int>>& negative_ids)
    {
        std::vector<int> positive_ids_vec, negative_ids_vec;
        if (negative_ids.has_value()) {
            negative_ids_vec = negative_ids.value();
        }
        else {
            negative_ids_vec = m_negative_ids_vec;
        }

        positive_ids_vec = positive_ids;
        if (positive_ids_vec.size() < seq_len) {
            positive_ids_vec.resize(seq_len, 0);
        }

        negative_ids_vec.insert(negative_ids_vec.end(), positive_ids_vec.begin(), positive_ids_vec.end());
        at::Tensor input_ids =
            torch::from_blob(negative_ids_vec.data(), {batch_size, seq_len}, get_options(torch::kInt32, torch::kCPU));
        at::Tensor attn_mask = input_ids != 0;

        m_input_ids.copy_(input_ids);
        m_attn_mask.copy_(attn_mask);

        m_engine.Call();

        return m_text_emb.clone();
    }

    at::Tensor get_attn_mask()
    {
        return m_attn_mask.clone();
    }
};

class UnetModel {
public:
    int batch_size;
    int height;
    int width;
    int emb_size;
    int seq_len;
    int emb_size_t5;
    int seq_len_t5;

private:
    libtrt::Engine m_engine;

    at::Tensor m_noisy_input;
    at::Tensor m_time_step;
    at::Tensor m_prompt_emb;
    at::Tensor m_attn_mask;
    at::Tensor m_prompt_t5_emb;
    at::Tensor m_attn_mask_t5;
    at::Tensor m_output;

public:
    UnetModel(const std::string& engine_path): m_engine(engine_path, 0, 0)
    {
        batch_size               = m_engine.GetInput(0)->GetDims(0);
        height                   = m_engine.GetInput(0)->GetDims(2);
        width                    = m_engine.GetInput(0)->GetDims(3);
        emb_size                 = m_engine.GetInput(2)->GetDims(2);
        seq_len                  = m_engine.GetInput(2)->GetDims(1);
        emb_size_t5              = m_engine.GetInput(4)->GetDims(2);
        seq_len_t5               = m_engine.GetInput(4)->GetDims(1);
        float* noisy_input_ptr   = reinterpret_cast<float*>(m_engine.GetInput(0)->GetPtr());
        int*   time_step_ptr     = reinterpret_cast<int*>(m_engine.GetInput(1)->GetPtr());
        float* prompt_emb_ptr    = reinterpret_cast<float*>(m_engine.GetInput(2)->GetPtr());
        bool*  attn_mask_ptr     = reinterpret_cast<bool*>(m_engine.GetInput(3)->GetPtr());
        float* prompt_t5_emb_ptr = reinterpret_cast<float*>(m_engine.GetInput(4)->GetPtr());
        bool*  attn_mask_t5_ptr  = reinterpret_cast<bool*>(m_engine.GetInput(5)->GetPtr());
        float* output_ptr        = reinterpret_cast<float*>(m_engine.GetOutput(0)->GetPtr());

        m_noisy_input =
            torch::from_blob(noisy_input_ptr, {batch_size, 4, height, width}, get_options(at::kFloat, at::kCUDA));
        m_time_step = torch::from_blob(time_step_ptr, {batch_size}, get_options(at::kInt, at::kCUDA));
        m_prompt_emb =
            torch::from_blob(prompt_emb_ptr, {batch_size, seq_len, emb_size}, get_options(at::kFloat, at::kCUDA));
        m_attn_mask     = torch::from_blob(attn_mask_ptr, {batch_size, seq_len}, get_options(at::kBool, at::kCUDA));
        m_prompt_t5_emb = torch::from_blob(
            prompt_t5_emb_ptr, {batch_size, seq_len_t5, emb_size_t5}, get_options(at::kFloat, at::kCUDA));
        m_attn_mask_t5 =
            torch::from_blob(attn_mask_t5_ptr, {batch_size, seq_len_t5}, get_options(at::kBool, at::kCUDA));
        m_output = torch::from_blob(output_ptr, {batch_size, 8, height, width}, get_options(at::kFloat, at::kCUDA));
    }

    at::Tensor forward(const at::Tensor& noisy_input,
                       const at::Tensor& time_step,
                       const at::Tensor& prompt_emb,
                       const at::Tensor& attn_mask,
                       const at::Tensor& prompt_t5_emb,
                       const at::Tensor& attn_mask_t5)
    {
        m_noisy_input.copy_(noisy_input);
        m_time_step.copy_(time_step);
        m_prompt_emb.copy_(prompt_emb);
        m_attn_mask.copy_(attn_mask);
        m_prompt_t5_emb.copy_(prompt_t5_emb);
        m_attn_mask_t5.copy_(attn_mask_t5);
        m_engine.Call();
        return m_output.clone();
    }
};

class VaeModel {
public:
    int batch_size;
    int height;
    int width;

    int new_height;
    int new_width;

private:
    libtrt::Engine m_engine;

    at::Tensor m_noisy_input;
    at::Tensor m_output;

public:
    VaeModel(const std::string& engine_path): m_engine(engine_path, 0, 0)
    {
        batch_size = m_engine.GetInput(0)->GetDims(0);
        height     = m_engine.GetInput(0)->GetDims(2);
        width      = m_engine.GetInput(0)->GetDims(3);
        new_height = m_engine.GetOutput(0)->GetDims(2);
        new_width  = m_engine.GetOutput(0)->GetDims(3);

        float* noisy_input_ptr = reinterpret_cast<float*>(m_engine.GetInput(0)->GetPtr());
        float* output_ptr      = reinterpret_cast<float*>(m_engine.GetOutput(0)->GetPtr());

        m_noisy_input =
            torch::from_blob(noisy_input_ptr, {batch_size, 4, height, width}, get_options(at::kFloat, at::kCUDA));
        m_output =
            torch::from_blob(output_ptr, {batch_size, 3, new_height, new_width}, get_options(at::kFloat, at::kCUDA));
    }

    at::Tensor forward(const at::Tensor& noisy_input)
    {
        m_noisy_input.copy_(noisy_input);
        m_engine.Call();
        return m_output.clone();
    }
};

class Pipeline {
public:
    int seed;
    int batch_size;
    int height;
    int width;
    int m_num_steps  = 100;
    int m_step_ratio = 10;

private:
    TextEmbeddingModel m_bert;
    TextEmbeddingModel m_t5;
    UnetModel          m_hunyuan;
    VaeModel           m_vae;
    at::Tensor         time_steps;

public:
    Pipeline(const std::string& bert_engine_file,
             const std::string& t5_engine_file,
             const std::string& hunyuan_engine_file,
             const std::string& vae_engine_file,
             int                seed = 42):
        m_bert(bert_engine_file),
        m_t5(t5_engine_file),
        m_hunyuan(hunyuan_engine_file),
        m_vae(vae_engine_file),
        seed(seed)
    {
        // '错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，'
        // clang-format off
        m_bert.set_default_negative_ids({101, 7231, 6428, 4638, 4706, 4714, 8024, 5136, 5130, 4638, 782, 5567, 8024, 3673, 2159, 8024, 5136, 5130, 4638, 5686, 3318, 8024, 1359, 2501, 8024, 1914, 865, 4638, 5501, 860, 8024, 3563, 5128, 4638, 7582, 5682, 8024, 3563, 5128, 8024, 7028, 1908, 8024, 4567, 2578, 8024, 3655, 5375, 8024, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        m_t5.set_default_negative_ids({259, 130401, 493, 132443, 261, 216210, 205958, 28936, 48211, 261, 113286, 20062, 261, 216210, 205958, 493, 43249, 261, 13969, 10473, 261, 3139, 16964, 493, 207490, 6667, 261, 28644, 158558, 493, 134193, 261, 28644, 158558, 261, 213134, 261, 14469, 44239, 261, 31761, 47193, 261, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        // clang-format on
        batch_size = m_hunyuan.batch_size;
        height     = m_hunyuan.height;
        width      = m_hunyuan.width;

        set_time_steps(m_num_steps);
        at::manual_seed(seed);
    }

    void set_time_steps(int num_steps)
    {
        m_num_steps    = num_steps;
        m_step_ratio   = 1000 / m_num_steps;
        at::Tensor tmp = at::arange(0, m_num_steps, 1, get_options(at::kInt, at::kCPU));
        tmp            = tmp.mul(m_step_ratio) + 1;
        tmp            = at::flip(tmp, {0});
        time_steps     = tmp.to(at::kCUDA);
    }

    py::array_t<uint8_t> run(const std::vector<int>&                bert_input_ids,
                             const std::vector<int>&                t5_input_ids,
                             const std::optional<std::vector<int>>& bert_neg_ids,
                             const std::optional<std::vector<int>>& t5_neg_ids,
                             int                                    num_steps = 100)
    {
        if (num_steps != m_num_steps) {
            set_time_steps(num_steps);
        }
        progressbar bar(m_num_steps);
        bar.set_todo_char(" ");
        bar.set_done_char("█");
        at::Tensor bert_emb       = m_bert.forward(bert_input_ids, bert_neg_ids);
        at::Tensor t5_emb         = m_t5.forward(t5_input_ids, t5_neg_ids);
        at::Tensor bert_attn_mask = m_bert.get_attn_mask();
        at::Tensor t5_attn_mask   = m_t5.get_attn_mask();
        at::Tensor noisy_input    = at::randn({1, 4, height, width}, get_options(at::kFloat, at::kCUDA));
        for (int i = 0; i < m_num_steps; ++i) {
            int step     = time_steps[i].item().toInt();
            int pre_step = step - m_step_ratio;
            pre_step     = pre_step >= 0 ? pre_step : 1;

            float alpha_prod_t      = alphas_cumprod[step];
            float alpha_prod_t_prev = alphas_cumprod[pre_step];

            float beta_prod_t      = 1.f - alpha_prod_t;
            float beta_prod_t_prev = 1.f - alpha_prod_t_prev;
            float current_alpha_t  = alpha_prod_t / alpha_prod_t_prev;
            float current_beta_t   = 1.f - current_alpha_t;

            at::Tensor latent_model_input = at::tile(noisy_input, {batch_size, 1, 1, 1});
            at::Tensor t_expand           = at::tensor({step, step}, get_options(at::kInt, at::kCUDA));
            at::Tensor latent_model_output =
                m_hunyuan.forward(latent_model_input, t_expand, bert_emb, bert_attn_mask, t5_emb, t5_attn_mask);
            at::Tensor noise_pred        = at::chunk(latent_model_output, 2, 1)[0];
            auto       chunk_tensor      = at::chunk(noise_pred, 2, 0);
            auto&      noise_pred_uncond = chunk_tensor[0];
            auto&      noise_pred_text   = chunk_tensor[1];
            noise_pred                   = noise_pred_uncond + 6.f * (noise_pred_text - noise_pred_uncond);

            at::Tensor pred_original_sample =
                std::sqrt(alpha_prod_t) * noisy_input - std::sqrt(beta_prod_t) * noise_pred;
            float pred_original_sample_coeff = std::sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t;
            float current_sample_coeff       = std::sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t;

            at::Tensor pred_prev_sample =
                pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_input;

            if (step > 0) {
                at::Tensor variance_noise = at::randn_like(pred_prev_sample);
                float      variance       = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t;
                variance                  = variance < 0 ? 0 : variance;
                variance_noise            = std::sqrt(variance) * variance_noise;
                pred_prev_sample          = pred_prev_sample + variance_noise;
            }
            noisy_input = pred_prev_sample;
            bar.update();
        }
        std::cout << "\n";

        noisy_input = noisy_input / 0.13025f;

        at::Tensor image = m_vae.forward(noisy_input);

        image = image / 2.f + 0.5f;
        image = image.clamp(0.f, 1.f);
        image = image * 255.f;
        image = image.round();
        image = image.to(at::kByte);
        image = image[0].permute({1, 2, 0}).contiguous();
        image = image.to(at::kCPU);
        return py::array(image.sizes(), image.data_ptr<uint8_t>());
    }
};

PYBIND11_MODULE(py_hunyuan_dit, m)
{
    m.doc() = R"(py_hunyuan_dit module provided by triple-mu!)";
    SET_ATTR_TO(m, "__version__", "0.0.1");
    SET_ATTR_TO(m, "__author__", "triple-mu");
    SET_ATTR_TO(m, "__email__", "gpu@163.com");
    SET_ATTR_TO(m, "__date__", "2024-05-17");
    SET_ATTR_TO(m, "__qq__", "3394101");

    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<const std::string&, const std::string&, const std::string&, const std::string&, int>(),
             py::arg("bert_engine_file"),
             py::arg("t5_engine_file"),
             py::arg("unet_engine_file"),
             py::arg("vae_engine_file"),
             py::arg("seed") = 0,
             "Initialize the pipeline")
        .def("generate",
             &Pipeline::run,
             py::arg("bert_input_ids"),
             py::arg("t5_input_ids"),
             py::arg("bert_neg_ids"),
             py::arg("t5_neg_ids"),
             py::arg("steps") = 100);
}