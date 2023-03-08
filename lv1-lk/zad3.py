def Average(List):
    return sum(List)/len(List)

List = []

while True:
    print("Unesi broj:")
    user_input = input()
    if(user_input =="Done"):
        break
    else:
        try:
            number = float(user_input)
            List.append(number)
        except:
            print("You didn't enter a number.")

print("Brojeva u listi je : ",len(List))
print("Srednja vrijednost brojeva liste je : ", Average(List))
print("Maksimalna vrijednost u listi je: ", max(List))
print("Minimalna vrijednost u listi je: ", min(List))
