// pages/index.tsx
'use client';
import { useState } from 'react';
import { ChakraProvider, Box, VStack, Input, Button, Text, Flex } from '@chakra-ui/react';
import axios from 'axios';

interface Message {
  type: 'user' | 'bot' | 'error';
  content: string;
  runId?: string;
}

export default function Home() {
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async () => {
    if (!question.trim()) return;

    setIsLoading(true);
    const newMessage: Message = { type: 'user', content: question };
    setChatHistory(prev => [...prev, newMessage]);

    try {
      const response = await axios.post('/api/ask', {
        text: question,
        group_id: 'user_group_1' // You might want to make this dynamic
      });

      const botMessage: Message = { 
        type: 'bot', 
        content: response.data.response,
        runId: response.data.run_id
      };
      setChatHistory(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = { type: 'error', content: 'An error occurred. Please try again.' };
      setChatHistory(prev => [...prev, errorMessage]);
    }

    setIsLoading(false);
    setQuestion('');
  };

  const handleFeedback = async (runId: string, score: number) => {
    try {
      await axios.post('/api/feedback', {
        run_id: runId,
        score: score
      });
      console.log('Feedback sent successfully');
    } catch (error) {
      console.error('Error sending feedback:', error);
    }
  };

  return (
    <ChakraProvider>
      <Box maxWidth="800px" margin="auto" p={5}>
        <VStack spacing={4} align="stretch">
          <Box bg="gray.100" p={4} borderRadius="md" height="400px" overflowY="auto">
            {chatHistory.map((message, index) => (
              <Box 
                key={index} 
                bg={message.type === 'user' ? 'blue.100' : message.type === 'bot' ? 'green.100' : 'red.100'} 
                p={2} 
                borderRadius="md" 
                mb={2}
              >
                <Text>{message.content}</Text>
                {message.type === 'bot' && message.runId && (
                  <Flex mt={2}>
                    <Button size="sm" onClick={() => handleFeedback(message.runId!, 1)} mr={2}>👍</Button>
                    <Button size="sm" onClick={() => handleFeedback(message.runId!, 0)}>👎</Button>
                  </Flex>
                )}
              </Box>
            ))}
          </Box>
          <Flex>
            <Input 
              value={question} 
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Type your question here..."
              mr={2}
            />
            <Button onClick={handleSubmit} isLoading={isLoading}>Send</Button>
          </Flex>
        </VStack>
      </Box>
    </ChakraProvider>
  );
}
