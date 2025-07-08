import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import TitleScreen from './TitleScreen';
import SurveyScreen from './SurveyScreen';
import MoodMeterScreen from './MoodMeterScreen';

import HighEnergyPleasant from './HighEnergyPleasant';
import HighEnergyUnpleasant from './HighEnergyUnpleasant';
import LowEnergyPleasant from './LowEnergyPleasant';
import LowEnergyUnpleasant from './LowEnergyUnpleasant';
const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Title">
        <Stack.Screen 
          name="Title" 
          component={TitleScreen} 
          options={{ headerShown: false }} 
        />
        <Stack.Screen name="Survey" component={SurveyScreen} />
        <Stack.Screen name="MoodMeter" component={MoodMeterScreen} />
		<Stack.Screen name="HighEnergyPleasant" component={HighEnergyPleasant} />
		<Stack.Screen name="HighEnergyUnpleasant" component={HighEnergyUnpleasant} />
		<Stack.Screen name="LowEnergyPleasant" component={LowEnergyPleasant} />
		<Stack.Screen name="LowEnergyUnpleasant" component={LowEnergyUnpleasant} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}