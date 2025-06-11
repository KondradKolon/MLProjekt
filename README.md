# MLProjekt

5
  
    
 
Grzegorz Madejski 
grzegorz.madejski@ug.edu.pl  
Zakład Sztucznej Inteligencji,  
Instytut Informatyki, Uniwersytet Gdański    
 
Strona:  1  
Aktualizacja: listopad 2024                                                                                                                                                                                                                                  
 
PROPOZCYJE PROJEKTÓW 2024/25 
INFORMATYKA – STUDIA II STOPNIA – NIESTACJ. 
C E L   P R O J E K T U  
Celem projektu jest wykorzystanie technik poznanych na zajęciach w większym, samodzielnym zadaniu. W szczególności 
chodzi o następujące techniki/algorytmy/modele: 
• Preprocessing, normalizacja, augmenacja danych 
• Klasyfikacja prostymi klasyfikatorami (NB, kNN, DecTree) 
• Klasyfikacja sieciami neuronowymi 
• Klasyfikacja obrazów konwolucyjnymi sieciami neuronowymi (i ewentualnie innymi modelam) 
• Transfer learning 
• Object Detection 
• Generatywne AI – grafika (GAN i inne) 
• Video / Audio / Tekst  
• Algorytmy analizy tekstu (bag of words, wyciąganie wydźwięku / opinii, tematu z tekstu) 
• Rekurencyjne sieci neuronowe (RNN, LSTM) oraz transformery lub inne modele deep learning 
• Algorytmy genetyczne 
• Inteligencja roju (PSO, ACO) 
 
Projekt można zrealizować na dwa sposoby: 
 
 
 
 
 
 
R E A L I Z A C J A   P R O J E K T U  
Kilka zasad realizacji projektu: 
• Projekt powinien dotyczyć tematyki wskazanej wyżej.  
• Projekt można realizować w dowolnym języku, Python nie jest wymagany. 
• Projekt realizujemy samodzielnie. W szczególnych przypadkach (np. spory stopień skomplikowania) można za 
zgodą prowadzącego realizować w parach. 
• Projekt powinien być unikalny dla każdej osoby (zespołu) w grupie. Prowadzący założy konwersację dla każdej 
grupy, gdzie należy rezerwować swój temat (kto pierwszy ten lepszy).  
• Należy projekty przechowywać na repozytorium Gitlab i dodać prowadzącego do repozytorium, 
 login gmadejski-ug    
• Czas na wykonanie projektu: do ostatnich zajęć. 
• Projekty będą rozliczane podczas ostatnich zajęć. Możliwe jest indywidualne prezentowanie rozwiązania 
prowadzącemu zajęcia, lub prezentacja na rzutniku przed grupą. Zaleca się, by przynajmniej parę osób 
zaprezentowało swój projekt publicznie. 
RAPORT BADAWCZY 
Dla wybranego problemu/bazy danych – 
sprawdzam jakie modele/techniki działają 
próbując osiągnąć jak najlepsze wyniki 
klasyfikacji. Robię porównanie wydajności w 
formie sprawozdania. 
 
APLIKACJA/SERWIS 
Stosuję wybrany, najlepszy model 
klasyfikujący w aplikacji/serwisie do 
praktycznych zastosowań. Prezentuję 
aplikację z jakąś prostą dokumentacją. 
  
    
 
Grzegorz Madejski 
grzegorz.madejski@ug.edu.pl  
Zakład Sztucznej Inteligencji,  
Instytut Informatyki, Uniwersytet Gdański    
 
Strona:  2  
Aktualizacja: listopad 2024                                                                                                                                                                                                                                  
 
O C E N A   P R O J E K T U  
Niestety w przypadku projektów o różnej tematyce ciężko wyznaczyć jednoznaczną miarę oceniania. Prowadzący zajęcia 
postarają się wystawić sprawiedliwe punkty, biorąc pod uwagę następujące kryteria: 
• Czy wkład studenta (czas, energia, własny kod programistyczny) w projekt był duży czy mały?  
o Tutaj warto, żeby student wskazał co jest zrealizowane samodzielnie, co skopiowane z samouczka, aco 
wygenerowane przez AI. Należy też uwzględnić wszystkie źródła, z których korzystaliśmy. 
• Czy projekt jest oryginalny/nowatorski, czy projekt jest raczej dobrze zbadany/odtwórczy? 
• Czy projekt jest dobrze zrealizowany (zawiera wszystkie istotne komponenty: preprocessing, algorytmy, modele, 
funkcjonalności)? 
• Czy projekt sięga po stare oklepane schematy, czy raczej student starał się korzystać z najnowocześniejszych 
technik, algorytmów udostępnianych w artykułach naukowych, blogach naukowych itp.? 
• Czy student był w stanie dobrze zaprezentować projekt? 
WYMAGANIA SZCZEGÓŁOWE 
Poniżej podane są wymagania szczegółowe. Student nie musi realizować wszystkich wymagań szczegółowych, ale są one 
dobrym wyznacznikiem czy projekt jest dobrze realizowany. 
RAPORT BADAWCZY APLIKACJA/SERWIS 
Baza danych 
Czy baza danych jest ciekawa/oryginalna/słabo zbadana? 
Czy samodzielnie ją stworzę, czy jest ściągnięta? 
Większość z Państwa pewnie ściągnie gotowca. Warto 
wybrać bazy danych duże (>10 000 próbek), nietrywialne 
(dużo kolumn, niełatwe obrazki), może też z błędami 
wymagającymi naprawy. 
Baza danych 
Trochę jak po lewej, ale wymagania są nieco mniejsze. 
Jeśli aplikacja jest prototypem, to właściwie baza danych 
może być ręcznie stworzona i nie musi być duża. Warto 
jednak skolekcjonować przynajmniej 100 próbek. 
Preprocessing 
Bazę danych należy naprawić (usuwanie brakujących 
danych, błedów, wartości odstających). Należy sprawdzić 
balans klas i ewentualnie zbalansować dane (imputacja, 
downsampling, upsampling, augemntacja obrazów).  
Preprocessing 
Warto rozpatrzyć kroki wymienione po lewej stronie. 
Klasyfikacja 
Należy porównać kilka algorytmów klasyfikujących, w tym 
sieci neuronowe. Warto je przebadać różnymi miarami 
(accuracy, confusion matrix, learning curve, itp.). 
Eksperymenty należy powtórzyć wielokrotnie i uśrednić 
wynik. Można stosować cross-validation. 
Klasyfikacja & Optymalizacja 
Właściwie wypada skupić się na klasyfikatorach, które 
najlepiej pasują do naszego zadania. Taki klasyfikator 
trzeba dobrze skonfigurować: powinien być nie tylko 
precyzyjny, ale również lekki. Czas obliczeń jest bardzo 
ważny, zwłaszcza na słabszych maszynach/aplikacjach 
mobilnych. 
Sprawozdanie 
Eksperymenty opisz w przejrzystym sprawozdaniu 
podzielonym na rozdziały. Można je zrobić w formie 
docx/pdf lub notatnika jupyterowego, albo nawet 
prezentacji powerpoint czy pliku latexowego.  
Sprawozdaniu powinny znajdować się Twoje objaśnienia, 
komentarze, wstawki kodu (najbardziej istotne), wykresy i 
grafiki, tabelki z wynikami. Dobrze będzie jeśli 
sprawozdanie rozpocznie się krótkim wstępem i zakończy 
konkluzjami podsumowującymi eksperymenty. Dołącz 
bibliografię z źródłami. 
Dokumentacja/Sprawozdanie 
Opisz wszystkie funkcjonalności aplikacji. 
Opisz eksperymenty tak jak powiedziano po lewej, ale 
może nie tak dokładnie jak w przypadku raportu.  
Możesz napisać instrukcję dla użytkownika aplikacji 
(readme). 
 
 
 
  
    
 
Grzegorz Madejski 
grzegorz.madejski@ug.edu.pl  
Zakład Sztucznej Inteligencji,  
Instytut Informatyki, Uniwersytet Gdański    
 
Strona:  3  
Aktualizacja: listopad 2024                                                                                                                                                                                                                                  
 
1 )   K L A S Y F I K A C JA   N A   D A T A S E C I E   N U M E R Y C ZN O - K A T E G OR I A L N Y M  
 
 
 
Jest to projekt typu standard, gdzie pobieramy lub tworzymy dataset w postaci 
tabelki, a następnie klasyfikujemy rekordy, tak jak robiliśmy to dla iris.csv lub 
diabetes.csv. 
 
Projekt należy zacząć od wybrania bazy danych. Jest tutaj kilka możliwości. 
Możemy wybrać dataset ze strony np. 
• https://www.kaggle.com/datasets wpisując odpowiednie słowa np. 
„classification” i „numeric” 
https://www.kaggle.com/datasets?search=classification+numeric  
• https://archive.ics.uci.edu/datasets  
 
Kilka dość oklepanych przykładów z Kaggle (były jako projekty w zeszłych 
latach): 
• https://www.kaggle.com/datasets/uciml/adult-census-income 
Rozpoznawanie ile zarabia osoba. 
• https://www.kaggle.com/datasets/blastchar/telco-customer-churn Klasyfikacja klientów telefonii komórkowej: 
czy przedłużą umowę, czy zrezygnują? 
• https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 Jaką jakoś ma czerwone wino? 
(można zmienić liczbę klas z 1-10 na good/bad). 
• https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset Diagnoza choroby serca. 
• https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data Diagnoza choroby tarczycy. 
• https://www.kaggle.com/datasets/uciml/mushroom-classification Klasyfikacja grzybów jadalnych i niejadalnych. 
Należy taką bazę danych przebadać pod katem klasyfikacji i wyniki przedstawić w czytelnej formie w formie raportu lub 
prezentacji. Można podzielić raport na rozdziały: 
• Baza danych: struktura, znaczenie kolumn, co jest klasyfikowane, jakie są wartości, wyświetlenie statystyk dla 
kolumn lub nawet wykresów. Objaśnienie. 
• Preprocessing: czy baza danych wymaga naprawy? Jakie są błędy? Czy są wartości odstające, brakujące? Jaki 
rozdzaj normalizacji lub modyfikacji danych działać będzie najlepiej? Czy zbiór trzeba balansować dla klas? Jeśli 
tak, to jaką metodą? 
• Klasyfikacja: testowanie różnych klasyfikatorów poznanych na zajęciach. Dla każdego klasyfikatora można 
rozpatrzyć różnorakie konfiguracje parametrów. Ewaluacja klasyfikatorów powinna być przeprowadzona za 
pomocą różnych sensownych miar. Można skorzystać z modeli pre-trained. 
• Reguły asocjacyjne: można poszukać w bazie danych jakichś ciekawych zależności. 
• Podsumowanie i interpretacja wyników: co wyszło, co nie wyszło? Co działa, a co nie? Czy są jakieś 
interesujące wnioski z badań? 
Im ciekawszy, bardziej szczegółowy i dociekliwy raport, tym lepsza ocena za niego. Będą brane takie aspekty jak: 
• Czy baza danych jest interesująca, oryginalna, stworzona własnoręcznie? 
• Czy na bazie danych dokonano jakiegoś istotnego preprocessingu? 
• Czy klasyfikatory zostały dostatecznie szczegółowo przebadane (pod katem parametrów, miar, wersji bazy 
danych, krzywych uczenia, itp.)? 
• Czy dodano reguły asocjacyjne? 
 
 
 
 
 
#klasyfikacja #numeryczno-kategorialne #uczenie-nadzorowane 
  
    
 
Grzegorz Madejski 
grzegorz.madejski@ug.edu.pl  
Zakład Sztucznej Inteligencji,  
Instytut Informatyki, Uniwersytet Gdański    
 
Strona:  4  
Aktualizacja: listopad 2024                                                                                                                                                                                                                                  
 
2 )   K L A S Y F I K A C JA   OB R A ZÓW  
 
 
 
W zasadzie opis będzie podobny do tego powyżej. Bazę danych można 
próbować tworzyć ręcznie lub pobrać z Kaggle. Kilka przykładów: 
• https://www.kaggle.com/datasets/gpiosenka/100-bird-species 
Klasyfikacja gatunków ptaków 
• https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-
classification   Klasyfikacja gatunków motyli 
• https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-
dataset   https://www.kaggle.com/datasets/kostastokis/simpsons-faces   
Klasyfikacja postaci z Simpsonów 
• https://www.kaggle.com/datasets/andyczhao/covidx-cxr2 Diagnoza COVID 
na podstawie zdjęcia RTG płuc 
• https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-
classification Klasyfikacja roślin po obrazie liścia 
Istnieje wiele innych: https://www.kaggle.com/search?q=image+classification+in%3Adatasets (polecam przejrzeć) 
Wymagania jak w propozycji A, choć preprocessing może wyglądać trochę inaczej (augmentacja zdjęć, kompresja, 
konwersja do szarości lub nie, filtry, itp.). 
Najlepszym klasyfikatorem będzie tutaj pewnie CNN z różnymi konfiguracjami. Ale warto też potestować inne 
klasyfikatory (zwykłe NN, kNN) i może różne modele transfer learning. 
Krzywe uczenia są tutaj mile widziane. 
W ramach dodatkowego bonusa do tego projektu, można spróbować wytrenować sieć GAN do generowania obrazów z 
badanej bazy danych. 
  
3 )   T W OR ZE N I E   OB R A Z ÓW   G E N E R A T Y W N Y M   A I  
 
 
 
Celem projektu jest przetestowanie różnych technik tworzenia grafiki za pomocą 
generatywnej AI. Na zajęciach poznaliśmy GAN. Chcemy rozszerzyć te badania o 
modele wykraczające poza materiał zajęć. 
 
Szczegółowy opis (wygenerowany przez ChatGPT – może mieć błędy) 
Celem projektu jest zapoznanie się z różnymi metodami generowania grafiki za 
pomocą modeli deep-learning. Studenci mają za zadanie wytrenować model 
generujący grafiki na podstawie wybranej bazy danych obrazów (np. z Kaggle) i 
porównać różne podejścia pod kątem jakości generowanych wyników, stabilności 
treningu i efektywności. 
Zakres projektu: 
1.  Pobranie i przygotowanie danych: 
o Studenci wybierają bazę danych grafik dostępnych na Kaggle (np. zdjęcia, 
obrazy rysunkowe, ikonki itp.). 
o Dokonują wstępnej analizy danych oraz ich przetwarzania (normalizacja, 
skalowanie, augmentacja danych). 
2. Implementacja modelu bazowego: 
o Jako podstawowy model sugeruje się implementację Generative Adversarial Network (GAN). 
o Studenci muszą zaimplementować dwa komponenty GAN: generator i dyskryminator, a następnie 
przetestować jego działanie. 
#klasyfikacja #obrazy #uczenie-nadzorowane 
#generatywne-ai #obrazy 
#deep-learning 
#deep-learning 
  
    
 
Grzegorz Madejski 
grzegorz.madejski@ug.edu.pl  
Zakład Sztucznej Inteligencji,  
Instytut Informatyki, Uniwersytet Gdański    
 
Strona:  5  
Aktualizacja: listopad 2024                                                                                                                                                                                                                                  
 
3. Testowanie alternatywnych metod: Studenci mają możliwość zaimplementowania innych modeli 
generatywnych, takich jak: 
o Variational Autoencoders (VAE): Model probabilistyczny uczący się rozkładu danych. Jest łatwiejszy w 
implementacji niż GAN i stabilniejszy w treningu. 
o Conditional GAN (cGAN): Rozszerzenie GAN, które pozwala na generowanie obrazów warunkowanych 
na dodatkowych etykietach (np. generowanie obrazów kotów, psów lub innych obiektów w zależności od 
klasy). 
o Deep Convolutional GAN (DCGAN): Usprawniona wersja GAN wykorzystująca splotowe warstwy 
neuronowe (Convolutional Neural Networks, CNN), co poprawia jakość generowanych obrazów. 
o StyleGAN: Zaawansowany model do generowania realistycznych obrazów z możliwością manipulacji 
stylami. 
o Denoising Diffusion Probabilistic Models (DDPM): Nowoczesny model generatywny oparty na procesie 
dyfuzji, pozwalający uzyskać wysoką jakość wyników. 
o Neural Radiance Fields (NeRF): Model do generowania grafiki 3D, który może być opcjonalnie 
zaimplementowany w bardziej zaawansowanej wersji projektu. 
4. Ewaluacja wyników: 
o Jakość generowanych obrazów może być oceniana za pomocą metryk takich jak: 
▪ Frechet Inception Distance (FID): mierzący podobieństwo dystrybucji wygenerowanych i 
prawdziwych obrazów. 
▪ Inception Score (IS): oceniający różnorodność i jakość wygenerowanych obrazów. 
o Można również przeprowadzić subiektywną ocenę jakości generowanych grafik przez zespół. 
5. Raport końcowy: 
o Opis wykorzystanych modeli. 
o Porównanie wyników różnych metod (jeśli studenci testowali więcej niż jedną). 
o Wnioski dotyczące jakości, stabilności i trudności implementacji różnych podejść. 
 
Przydatne linki 
1. Generative Adversarial Networks (GAN): 
o Artykuł przeglądowy: https://arxiv.org/abs/2111.13282 
o Kurs na Hugging Face: https://huggingface.co/learn/computer-vision-course/en/unit5/generative-
models/gans  
2. Variational Autoencoders (VAE): 
o Wykład "Variational Autoencoders": https://www.cs.columbia.edu/~zemel/Class/Nndl-
2021/files/lec13.pdf  
3. Porównanie VAE i GAN: 
o Artykuł przeglądowy: https://arxiv.org/abs/2103.04922 
4. StyleGAN: 
o Artykuł na temat StyleGAN: https://en.wikipedia.org/wiki/StyleGAN 
5. Denoising Diffusion Probabilistic Models (DDPM): 
o Artykuł przeglądowy: https://en.wikipedia.org/wiki/Diffusion_model 
6. Neural Radiance Fields (NeRF): 
o Artykuł przeglądowy: https://link.springer.com/chapter/10.1007/978-981-97-2550-2_23 
 
4 )   W Y K R Y W A N I E   OB I E K T ÓW   N A   W I D E O  I   S T E R OW A N I E  
 
 
 
Fajnie by było grać w grę nie klikając w myszkę, komórkę, klawiaturę czy konsolę, a 
wyginając ciało śmiało          
Super film na zajawkę o co chodzi: 
https://www.youtube.com/watch?v=Vi3Li3TkUVY&ab_channel=EverythingIsHacked  
Celem projektu byłoby stworzenie appki, a właściwie jej prototypu – wystarczy 
zgrubnie napisany program, która będzie jakąś grą (ściągniętą lub napisaną 
własnoręcznie) z możliwością sterowania obiektem w grze za pomocą ruchu. 
O jakich ruchach mowa? Mogą to być gest dłonią (trochę łatwiejsze), machanie 
jakimś przedmiotem ze znacznikami, poruszanie całym ciałem. Można dla ułatwienia 
#obrazy #wideo #sterowanie #deep-learning 
  
    
 
Grzegorz Madejski 
grzegorz.madejski@ug.edu.pl  
Zakład Sztucznej Inteligencji,  
Instytut Informatyki, Uniwersytet Gdański    
 
Strona:  6  
Aktualizacja: listopad 2024                                                                                                                                                                                                                                  
 
założyć, że osoba jest w odpowiedniej odległości lub odpowiednio ubrana. Projekt jest ciekawy, ale trzeba nałożyć sobie  
parę zdroworozsądkowych ułatwień. 
Można przetestować kilka opcji (tak jak ten gość z filmiku) – jeśli nie uda się z machaniem rękami, to może skupmy się na 
dłoniach? Jeśli nasz własny model się nie wytrenował to  może zastosujmy transfer learning? Kilka linków do przejrzenia: 
• https://huggingface.co/datasets/sayakpaul/poses-controlnet-dataset  
• https://huggingface.co/spaces/hysts/mediapipe-pose-estimation  
• https://developers.google.com/mediapipe + https://github.com/google/mediapipe  
 
5) INŻYNIERIA PROMPTÓW I PRZEGLĄD NARZĘDZI DO TWORZENIA WIDEO 
 
 
 
Tematem projektu jest przejrzenie narzędzi generatywnego AI do tworzenia 
filmów, porównanie ich i wybranie najlepszych narzędzi do stworzenia krótkiego 
filmu. W ramach tego projektu należy stworzyć dwie rzeczy: 
• Krótki raport (które narzędzie testowaliśmy? Jakie były wygenerowane 
rezultaty? Czy spełniły nasze oczekiwania? Dlaczego tak/nie? Jakie prompty były 
stosowane do naszego filmu? Opisz krok po kroku jak był tworzony film.) 
• Film krótkometrażowy na wybrany temat. Może to być krótki film fabularny, 
albo promocyjny (reklama), animowana bajka, itp. 
Inspiracje: 
a. https://www.facebook.com/groups/846203050725189 
b. https://www.facebook.com/groups/929232341613889  
 
Tworzenie filmu to proces dość skomplikowany. Film trzeba złożyć z kilku 
komponentów: wideo + audio (muzyka) + audio (dialog). Będzie dobrze jeśli 
wszystkie trzy komponenty pojawią się w naszym filmie. 
 
1) Wideo.  
Wykorzystaj narzędzie do tworzenia wideo. Przykładowe narzędzia (nie jest to wyczerpująca lista):  
• Runway ML : https://runwayml.com/ 
• Minimax / Hailuo AI: https://hailuoai.video/  
• Vidu: https://vidu.studio/  
• Kling AI: https://klingai.com/  
• Luma Labs: https://lumalabs.ai/dream-machine  
• Haiper AI: https://haiper.ai/  
• Na Huggingface są też modele do generowania filmów, np. CogVideoX wyszedł ostatnio i jest ponoć dobry. 
Wymaga to jednak ściągnięcia dużego (kilka GB) modelu na komputer, odpalenia i czekania aż film się 
wygeneruje. Trzeba mieć dobrą maszynę. Istnieje wiele narzędzi do automatyzowania procesu generowania 
filmu, np. ComfyUI czy DreamBooth. Osoby bardziej ambitne mogą spróbować się z nimi zmierzyć.  
• Inne, wyżej nie wspomniane? Poszukaj w necie. 
Kilka podpowiedzi: 
• Filmy generować można za pomocą prompta tekstowego (Text-to-Video), albo za pomocę prompta 
tekstowego z obrazkiem, który staje się pierwszą klatką filmu: (Image-to-Video). Należy się zastanowić, która 
metoda nas bardziej interesuje. Są fajne narzędzie do generowania obrazów, które mogą nam taką peirwszą 
klatkę filmu wygenerować i często widać ludzi którzy generują obraz w Midjourney, a potem używają innego 
narzędzia do generowania wideo np. Minimax. 
• Narzędzia te nie wygenerują pewnie całego filmu. Trzeba generować scena po scenie. Z reguły narzędzia 
oferują tylko kilkusekundowe filmy. Jeśli chcemy wygenerować dłuższą scenę, musimy wydłużyć początkowy 
film. Jeśli narzędzie ma taką opcji, można skopiować ostatnią klatkę pierwszego wygenerowanego filmu i na jej 
bazie odpalić Image-To-Video i dogenerować resztę sceny. 
• Problemem może być niezapamiętywanie obiektu między scenami. W jednej scenie nasz bohater może mieć 
czapkę, a w drugiej ona mu zniknie (albo zmieni kolor). Jeszcze gorzej jest jak twarz się zmienia 😉 Tutaj trzeba 
skorzystać z narzędzi typu „daj referencję twarzy / wyglądu” i generuj film pamiętając tę twarz. Niektóre 
narzędzie takie coś oferują. Inny sposób: generować tak długo (powtarzać próby z doprecyzowanie 
szczegółów wyglądu) aż bohater w kolejnej scenie upodobni się do tego z pierwszej sceny. 
#generatywne-ai #obrazy #wideo #audio #prompty #deep-learning 
  
    
 
Grzegorz Madejski 
grzegorz.madejski@ug.edu.pl  
Zakład Sztucznej Inteligencji,  
Instytut Informatyki, Uniwersytet Gdański    
 
Strona:  7  
Aktualizacja: listopad 2024                                                                                                                                                                                                                                  
 
 
2) Audio: mowa. Wykorzystaj narzędzie do generowania głosu (voice generator) np. https://elevenlabs.io/ (są też 
inne które można przetestować: https://zapier.com/blog/best-ai-voice-generator/ , https://dorik.com/blog/best-
ai-voice-generators ). Narzędzie to, powinno wygenerować track audio do naszego filmu, z rozmową pomiędzy 
bohaterami, lub (trochę łatwiejsza opcja) z głosem narratora opowiadającym historię. W przypadku pierwszej i 
trudniejszej opcji (tzn. dialogu miedzy bohaterami), można zadbać o to, żeby twarze/usta poruszały się zgodnie z 
wygłaszanym tekstem (sprawdź czy jest możliwość synchronizacji głosu z wideo).  
 
3) Audio: muzyka. Dodaj pasujący do filmu podkład muzyczny wygenerowany za pomocą AI.  
Wykorzystaj narzędzie do tworzenia muzyki np. 
• https://www.udio.com/  
• https://suno.com/  
 
Wszystkie trzy komponenty należy złożyć jakimś programem do edycji wideo. Tutaj nie podpowiadam, proszę znaleźć 
dobre narzędzie. Idealnie by było gdyby filmik miał przynajmniej parę scen i minimum 30 sekund. Może komuś uda się 
złożyć nawet minutę lub więcej? 😉 
 
 
6) INŻYNIERIA PROMPTÓW I CHATBOT 
 
 
 
Celem projektu jest wykorzystanie LLM (large language model), do generowania 
odpowiedzi w specyficznym kontekście/wirtualnym środowisku. Przykłady: 
• Chcemy czatbota na stronę Inf.ug.edu.pl i chatbot powinien znać wszystkie 
sylabusy i odpowiadać studentom lub kandydatom na studia na pytania na 
temat studiów. 
Użytkownik: „Co przerabiane jest na przedmiocie Inteligencja obliczeniowa?” 
Chatbot (LLM): „Zgodnie z sylabusem dla kierunki Informatyka, na Inteligencji 
obliczeniowej przerabiane jest uczenie maszynowe, algorytmy 
metaheurystyczne, deep learning” 
Użytkownik: „Możesz wyjaśnić co to za pojęcia” 
Chatbot (LLM): „Tak, oto wyjaśnienie: ...” 
• Chcemy postać w grze (NPC, non-playable character) typu RPG, która będzie z 
nami miała interakcję lepszą niż tylko prosty dialog typu „wybierz odpowiedź a 
lub b”. Chcemy, żeby postać z nami rozmawiała swobodnie, ale żeby dialog 
pasował do świata przykład: 
Gracz: „Kowalu, czy gdzie w tym mieście jest karczma?” 
NPC (LLM): „Musisz iść na północ, mości panie. Karczma jest za młynem.” 
Gracz: „Dzięki, ziomal. Lubisz grać na kompie?” 
NPC (LLM): „Cóż mówisz, wędrowcze. Nie rozumiem tych przedziwnych słów!” 
     Kolejna rozmowa można nawiązywać do poprzedniej, o ile jest zapamiętana: 
    NPC (LLM): „O znowu przybywasz. Mów tym razem bardziej zrozumiale, mości panie” 
Projekt można zrobić naprawdę ambitnie, testując różne metody i robiąć wszystko „od podszewki”. Można też skorzystać 
z gotowych skryptów i rozwiązań, co będzie sporym ułatwieniem. Jeśli ktoś podejdzie do tego ambitniej, możemy się 
umówić, ze ten projekt zaliczy oba projekty. W przypadku sporego zaangażowania, można pewnie to rozwinąć w kierunku 
pracy mgr lub napisać raport naukowy (w czym mogę pomóc i się trochę zaangażować). 
 
Przykłady Projektów i Technik z linkami (podpowiedzi od ChatGPT 😉, warto dokładniej przeszukać internet). 
1. Dynamiczna generacja dialogów 
Modele LLM, takie jak GPT-3 i GPT-4, mogą tworzyć dialogi kontekstowe zamiast korzystać z sztywnych linii 
dialogowych. Dzięki temu NPC są bardziej naturalni i immersyjni. 
https://lutpub.lut.fi/bitstream/handle/10024/167809/bachelorthesis_Huang_Junyang.pdf?sequence=1  
2. Dialogi świadome kontekstu 
NPC generują odpowiedzi uwzględniające stan gry lub wcześniejsze interakcje z graczem, co zwiększa 
zaangażowanie. 
https://projekter.aau.dk/projekter/files/536738243/The_Effect_of_Context_aware_LLM_based_NPC_Dialogues_o
n_Player_Engagement_in_Role_playing_Video_Games.pdf  
#generatywne-ai #obrazy #wideo #audio #prompty 
