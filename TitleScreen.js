import React from 'react';
import { View, Text, ImageBackground, TouchableOpacity, StyleSheet } from 'react-native';

const TitleScreen = ({ navigation }) => {
  return (
    <ImageBackground 
      source={require('./assets/iPhone 16 - 1.png')} // Ensure the image is placed in the 'assets' folder
      style={styles.background}
      resizeMode="cover"
    >
      <View style={styles.container}>
        <Text style={styles.title}>MoodMeter</Text>
        <TouchableOpacity 
          style={styles.button} 
          onPress={() => navigation.navigate('Survey')}
        >
          <Text style={styles.buttonText}>START</Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  background: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
  },
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0)',
    padding: 20,
    borderRadius: 10,
  },
  title: {
    fontSize: 65,
    fontWeight: 'bold',
  },
  button: {
    marginTop: 20,
    backgroundColor: 'black',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 5,
  },
  buttonText: {
    color: 'white',
    fontSize: 30,
    fontWeight: 'bold',
  },
});

export default TitleScreen;
